# 26c49f13 到 fb26de78 变更总结

## 1. 统计范围

- 基线 commit：`26c49f13d25b34ade18565a10fbab30078eabc8e`
- 基线时间：2025-12-01
- 截止 commit：`fb26de789433e153433dec5515bf0d8d03b8f042`
- 截止时间：2026-01-06
- 提交统计口径：`git log 26c49f13..fb26de78`
- 净变更统计口径：`git diff 26c49f13 fb26de78`
- 说明：以下统计不包含基线 commit 自身的改动，只统计它之后直到截止 commit 为止的变化

## 2. 总览

这一段历史可以概括为：项目从一个保留原始 `Sarathi` 调度器说明和基础能力的分支，逐步演化成面向 `Qwen/Qwen3`、`Opt-Sarathi` 调度、离线压测和阈值调参的研究型版本。

最终净变更为：

- 68 个提交
- 38 个文件发生最终净变化
- 2676 行新增
- 640 行删除

最终落地的主线变化有四条：

1. 新增 `OPT_SARATHI` 调度器及其配置、注册、调度策略和阈值控制能力。
2. 新增 `Qwen3` 模型支持，并重写 `Qwen` 权重加载逻辑以修复多卡运行问题。
3. 新增离线测试、输出持久化、批次特征采样 CSV、到达时间模拟和阈值选择脚本。
4. 将仓库文档、示例和默认配置调整为更贴近当前研究分支的使用方式。

## 3. 详细变更

### 3.1 调度器与调度策略

#### 3.1.1 新增 `OPT_SARATHI` 调度体系

- 在 `sarathi/types.py` 中新增 `SchedulerType.OPT_SARATHI`。
- 在 `sarathi/core/scheduler/scheduler_registry.py` 中注册 `OptSarathiScheduler`。
- 在 `sarathi/core/block_space_manager/block_space_manager_registry.py` 中注册 `OptSarathiBlockSpaceManager`。
- 新增 `sarathi/core/scheduler/opt_sarathi_scheduler.py`，形成新的调度实现。
- 新增 `sarathi/core/block_space_manager/opt_sarathi_block_space_manager.py`，当前实现继承自 `VLLMBlockSpaceManager`。

`OptSarathiScheduler` 的关键行为如下：

- 沿用分块 prefill 思路，但显式支持 `min_chunk_threshold`。
- 调度顺序分成三段：
  - 先处理已完成 prefill 的 decode 请求。
  - 再处理还未完成 prefill 的 running 请求。
  - 最后尝试吸纳 waiting 队列中的新请求。
- 当 decode 追加 token 时，如果 KV block 不足，会按策略抢占低优先级请求。
- 当本轮剩余 budget 小于 `min_chunk_threshold` 时，不再继续塞入新的 prefill chunk。
- 支持动态 chunk schedule，沿用 `low_chunk_size / high_chunk_size / chunk_schedule_max_tokens / chunk_schedule_stages` 组合。

#### 3.1.2 调度配置的扩展与默认值调整

`sarathi/config/config.py` 的变化比较集中：

- `ParallelConfig.pipeline_parallel_size` 默认值从 `2` 改成 `1`。
- `BaseSchedulerConfig` 新增 `policy_name`，允许从配置侧选择调度策略。
- `BaseSchedulerConfig.__post_init__` 增加策略合法性校验。
- 新增 `OptSarathiSchedulerConfig`，包含以下参数：
  - `chunk_size`
  - `enable_dynamic_chunking_schedule`
  - `low_chunk_size`
  - `high_chunk_size`
  - `chunk_schedule_max_tokens`
  - `chunk_schedule_stages`
  - `min_chunk_threshold`
  - `enable_select_stats_csv`
- `OptSarathiSchedulerConfig` 会校验 `min_chunk_threshold < chunk_size * 0.3`。
- `SystemConfig` 和 `BaseEndpointConfig` 的默认调度器从 `SarathiSchedulerConfig` 改为 `OptSarathiSchedulerConfig`。

这意味着在该范围结束时，项目默认已经把 `Opt-Sarathi` 当作主调度器入口。

#### 3.1.3 调度策略从单一 FCFS 扩展为多策略

`sarathi/core/policy.py` 被改造成一个真正的策略工厂：

- `Policy` 从普通类改为抽象基类。
- 保留 `FCFS`。
- 新增 `SPF`，按 prompt 长度优先短 prompt。
- 新增 `SRTF`，按剩余 prompt 长度优先。
- 新增 `AGING`，按等待时间和剩余 prompt 的加权和排序。
- `PolicyFactory` 新增 `get_available_policies()`，用于配置校验和帮助文本。

`AGING` 的最终公式为：

`priority = 125.0 * 0.3 * waiting_time - 1.0 * remaining_prompt_tokens`

这条线上还经历了两次调参：

- 2025-12-31：先把 `time_weight` 调整到 `588.0 * 0.3`
- 2026-01-06：又改回到最终版本 `125.0 * 0.3`

#### 3.1.4 `BaseScheduler` 对策略的接入方式改变

`sarathi/core/scheduler/base_scheduler.py` 的变化有两个关键点：

- 不再固定使用 `fcfs`，而是从 `scheduler_config.policy_name` 实例化策略。
- 当 pipeline 已满、当前轮无法继续调度新 batch 时，仍会对 `waiting` 队列按策略重新排序。

后一条改动很重要，因为它让 `AGING` 这类依赖等待时间变化的策略在多卡/流水线场景下仍能持续生效，而不是在 pipeline 满载期间完全失去更新机会。

### 3.2 模型适配与执行链路

#### 3.2.1 新增 `Qwen3` 模型支持

这部分新增了两个文件：

- `sarathi/model_executor/models/qwen3.py`
- `sarathi/transformers_utils/configs/qwen3.py`

对应能力包括：

- 新增 `Qwen3Config`
- 在 `model_loader.py` 中把 `Qwen3ForCausalLM` 注册到 `_MODEL_REGISTRY`
- 实现 `Qwen3` 的 attention、MLP、decoder layer、pipeline/tensor parallel 兼容路径
- 支持把 HuggingFace 风格权重映射到本项目内部的并行层实现
- 在权重加载时合并：
  - `gate_proj + up_proj -> gate_up_proj`
  - `q_proj + k_proj + v_proj -> qkv_proj`
- 支持 `Q-Norm`、`K-Norm`、RoPE、GQA 相关参数

从结果上看，这一段历史把项目的适配重点从原始模型族扩展到了 `Qwen3`。

#### 3.2.2 重写 `Qwen` 权重加载逻辑，修复多卡问题

这部分是 2025-12-31 的集中修改：

- 原有 `sarathi/model_executor/models/qwen.py` 被备份为 `qwen.py.bak`
- 新的 `qwen.py` 被重新创建

新版本 `qwen.py` 做了以下修复：

- 支持同时识别 `transformer.h.<id>.` 和 `model.h.<id>.` 两类层命名。
- 在 pipeline parallel 下，把全局层号映射为当前 stage 的局部层号。
- 非第一段 pipeline stage 不再错误加载 embedding。
- 非最后一段 pipeline stage 不再错误加载 `lm_head` 和 `ln_f`。
- 对 `c_attn` 的 QKV 融合权重进行正确的 tensor parallel 切片。
- 对 MLP 的 `w1/w2` 权重做拼接装配，映射到 `gate_up_proj`。
- 当参数名在本地 `state_dict` 中找不到时，抛出更清晰的错误信息，方便定位命名错配。

这组修改与两个连续提交绑定：

- `40bca37`：把旧 `qwen.py` 迁移为 `qwen.py.bak`
- `e89fdff`：创建新的 `qwen.py`

最终效果是：`Qwen` 在多卡环境下的权重装载与 stage 切分逻辑被显式修复。

#### 3.2.3 模型执行链路增加特征采样 CSV

`sarathi/model_executor/model_runner.py` 除了兼容 `OPT_SARATHI` 的 profile 外，还新增了一条用于分析调参的数据采样路径。

新增内容包括：

- `run_with_select_csv()` 执行入口
- `_append_select_stats_csv_row()` 写 CSV 辅助函数
- 在单个 batch 执行前后统计以下字段：
  - `decode_tokens`
  - `decode_history_tokens`
  - `prefill_tokens`
  - `prefill_processed_tokens`
  - `latency_ms`
- CSV 输出路径固定在当前 replica 输出目录下，文件名为 `select_stats_rank{rank}.csv`

同时：

- `BaseWorker.execute_model()` 增加条件分支
- 当调度器类型是 `OPT_SARATHI` 且 `enable_select_stats_csv=True` 时，走 `run_with_select_csv()`
- 否则仍走原来的 `run()`

这使得阈值选择脚本和批次分析可以直接消费真实 batch 特征，而不是只依赖最终请求级指标。

### 3.3 离线测试、数据处理与调参工具

#### 3.3.1 新增 prompt 数据提取与到达时间模拟

新文件 `sarathi/utils/prompt_utils.py` 提供了三类能力：

- 从 ShareGPT 格式 JSON 中提取每个样本的第一条 human prompt
- 按固定间隔生成平滑到达时间序列
- 生成聚集式到达时间序列，用于模拟突发请求

这让离线推理脚本不再局限于手写 prompt，而可以直接从数据集抽样并控制请求到达模式。

#### 3.3.2 新增输出持久化工具

新文件 `sarathi/utils/output_utils.py` 提供：

- 将生成结果打印到终端
- 将 `prompt` 与 `generated_text` 落盘为 CSV
- 自动创建 `offline_inference_output`
- 自动给文件名加时间戳

这条能力对应了 2025-12-10 之后关于“可持久化输出”和 2025-12-22 关于“CSV 文件名时间戳”的一系列提交。

#### 3.3.3 新增离线推理示例

新增：

- `example/chat_only.py`
- `example/offline_inference.py`

删除：

- `examples/offline_inference.py`

新的示例脚本相较旧版，变化方向非常明确：

- 模型切到 `Qwen/Qwen3-8B`
- 调度器切到 `OptSarathiSchedulerConfig`
- 示例从演示型 prompt 改成更偏离线评测与实验复现的脚本
- `offline_inference.py` 支持从数据集提取 prompt，并接收外部到达时间序列
- 生成长度和并发参数围绕消费级显卡场景做了收敛

其中一些后续增量包括：

- 2025-12-17：示例中的 `max_tokens` 改到 `2048`
- 2025-12-17：离线示例的模型参数调整为更适配消费级显卡
- 2025-12-31：离线示例的 `max_num_seqs` 从 `10` 提高到 `32`

#### 3.3.4 新增阈值调参与分析脚本

新增：

- `example/threshold/offline_get_threshold.py`
- `example/threshold/offline_get_threshold.sh`
- `example/threshold/select_threshold.py`
- `example/README.md`

这部分能力最后收敛成完整的阈值实验闭环：

- 用 `offline_get_threshold.py` 执行一次指定 `min_chunk_threshold` 的离线压测
- 用 `offline_get_threshold.sh` 从 `0` 到 `MAX_THRESHOLD` 批量遍历阈值
- 自动收集 `replica_0/plots` 下的关键 CSV
- 用 `select_threshold.py` 读取多次 run 的 CSV，先按基线阈值归一化，再按均值、中位数、P90 或最大值聚合，最后输出推荐阈值

纳入分析的核心指标包括：

- `request_execution_time` 的 P95
- `request_execution_time_normalized` 的均值
- `prefill_e2e_time` 的 P95
- `prefill_time_execution_plus_preemption_normalized` 的均值
- `decode_time_execution_plus_preemption_normalized` 的均值

需要注意的是，这条线中间经历过几轮命名和逻辑迭代，提交消息里曾出现“max chunk threshold”等表述，但最终保留下来的实现已经统一到 `min_chunk_threshold`。

### 3.4 文档、仓库结构与默认使用方式

#### 3.4.1 README 从原项目说明转向当前分支说明

`README.md` 的变化不是简单润色，而是整体定位切换：

- 标题从 `Sarathi-Serve` 改为 `Opt-Sarathi-Serve`
- 介绍文本切换成中文说明
- 补充了当前分支适配的模型与测试数据集
- 增加简单对话、离线测试、OpenAI 服务入口三类运行方式
- 把调参脚本入口引到 `example/README.md`
- 把项目定位明确为基于原始 `sarathi-serve` 的研究分支

#### 3.4.2 示例目录重组

仓库把面向当前分支的脚本统一放到了 `example/` 下，并新增了对应 README。原来的 `examples/offline_inference.py` 被删除。

#### 3.4.3 仓库治理文件被裁剪

在最终净变更中，以下文件被删除：

- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/workflows/codeql.yml`
- `.github/workflows/lint.yml`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`

从提交历史看，这部分中间伴随多次 `delete / revert / reapply`，但最终状态是这些治理和 CI 辅助文件被整体移除，仓库更偏研究代码分支，而不是面向公开协作的完整工程模板。

#### 3.4.4 其他小型结构调整

- `.gitignore` 多次修改，最终保留了若干本地化忽略规则。
- `sarathi/utils/base_registry.py`、`sarathi/core/datatypes/sequence.py`、`sarathi/core/block_space_manager/base_block_space_manager.py` 等文件主要是注释整理和轻微清理，逻辑变化较少。

## 4. 受影响文件清单

### 4.1 新增文件

- `example/README.md`
- `example/chat_only.py`
- `example/offline_inference.py`
- `example/threshold/offline_get_threshold.py`
- `example/threshold/offline_get_threshold.sh`
- `example/threshold/select_threshold.py`
- `sarathi/core/block_space_manager/opt_sarathi_block_space_manager.py`
- `sarathi/core/scheduler/opt_sarathi_scheduler.py`
- `sarathi/model_executor/models/qwen.py.bak`
- `sarathi/model_executor/models/qwen3.py`
- `sarathi/transformers_utils/configs/qwen3.py`
- `sarathi/utils/output_utils.py`
- `sarathi/utils/prompt_utils.py`

### 4.2 修改文件

- `.gitignore`
- `README.md`
- `sarathi/config/config.py`
- `sarathi/core/block_space_manager/base_block_space_manager.py`
- `sarathi/core/block_space_manager/block_space_manager_registry.py`
- `sarathi/core/datatypes/sequence.py`
- `sarathi/core/policy.py`
- `sarathi/core/scheduler/base_scheduler.py`
- `sarathi/core/scheduler/sarathi_scheduler.py`
- `sarathi/core/scheduler/scheduler_registry.py`
- `sarathi/engine/base_llm_engine.py`
- `sarathi/model_executor/model_loader.py`
- `sarathi/model_executor/model_runner.py`
- `sarathi/model_executor/models/qwen.py`
- `sarathi/types.py`
- `sarathi/utils/base_registry.py`
- `sarathi/worker/base_worker.py`

### 4.3 删除文件

- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/workflows/codeql.yml`
- `.github/workflows/lint.yml`
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `SUPPORT.md`
- `examples/offline_inference.py`

## 5. 按时间段梳理演进脉络

### 2025-12-03 到 2025-12-10

- 在原有算法注释基础上开始引入 `opt_sarathi`
- 把默认 pipeline 并行数改为 1
- 初步引入 `Qwen3` 模型支持
- 增加从数据集提取 prompt 的能力
- 加入 benchmark questions 和离线测试样例
- 新增 `SPF` 策略
- 增加输出持久化和 prompt 持久化方向的支持

### 2025-12-12 到 2025-12-19

- 处理 `.gitignore`、`.github` 等仓库治理文件
- 加入 `AGING` 与 `SRTF`
- 初始化 `example/` 目录和阈值实验脚本
- 调整示例参数，面向消费级显卡
- 增加对 batch 特征 CSV 的记录
- README 与说明文档改为更贴近当前分支的内容

### 2025-12-22 到 2025-12-26

- 为 CSV 输出名增加时间戳并修正落盘逻辑
- 新增离线测试中模拟请求到达时间的方法
- 继续优化 `AGING` 的参数选择思路
- 在 CSV 中增加 `decode_history_tokens`
- 删除旧的实验脚本残留

### 2025-12-31 到 2026-01-06

- 通过 `qwen.py.bak -> 新 qwen.py` 的方式修复 `Qwen` 多卡问题
- 让 `AGING` 在多卡/流水线场景下保持优先级更新
- 离线示例的 `max_num_seqs` 提高到 32
- 两次调整 `AGING` 的 `time_weight`
- 最终在 `fb26de7` 落定当前策略参数

## 6. 完整提交清单

以下列表按时间正序排列：

1. `ea6ad85` | 2025-12-03 | Charmstok | 添加注释，对应文算法三
2. `452d4f2` | 2025-12-05 | Charmstok | 修改默认并行 GPU 数为 1
3. `d9ef3b0` | 2025-12-05 | CharmsTok | 合并分支 PR #1
4. `dffcc7f` | 2025-12-05 | CharmsTok | 回滚一次实验性提交
5. `8132b80` | 2025-12-05 | CharmsTok | 合并回滚分支 PR #2
6. `fff23f1` | 2025-12-05 | Charmstok | 初次引入 `opt_sarathi`
7. `df1ca98` | 2025-12-08 | Charmstok | 移植 `Qwen3` 模型
8. `62c9178` | 2025-12-08 | Charmstok | 增加从数据集提取 prompts 的能力
9. `a4c1d23` | 2025-12-08 | Charmstok | 为 `opt_sarathi` 调度器补注释并修改 README
10. `858860a` | 2025-12-08 | Charmstok | 初始化补充
11. `2c16986` | 2025-12-09 | lhx-666-cool | 增加问题集
12. `ae6a236` | 2025-12-09 | CharmsTok | 增加 benchmark questions
13. `6d6f3af` | 2025-12-09 | Charmstok | 增加 `Qwen3` 离线测试样例
14. `d814f41` | 2025-12-09 | CharmsTok | 合并优化分支 PR #4
15. `8f39558` | 2025-12-09 | CharmsTok | 删除 `first_human_questions.txt`
16. `49685d1` | 2025-12-09 | Charmstok | 新增 `SPF` 调度策略
17. `37f9c1e` | 2025-12-09 | CharmsTok | 合并优化分支 PR #5
18. `d90c80a` | 2025-12-10 | Charmstok | 增加输出与提示词持久化能力
19. `029a659` | 2025-12-10 | CharmsTok | 合并优化分支 PR #6
20. `2d8db2d` | 2025-12-10 | Charmstok | 修正
21. `72711f6` | 2025-12-12 | Charmstok | 修正 12.12
22. `1f43ed1` | 2025-12-12 | CharmsTok | 修改 `.gitignore`
23. `29d9c55` | 2025-12-12 | CharmsTok | 修改 `.gitignore`
24. `1121d7f` | 2025-12-12 | CharmsTok | 删除 `.github`
25. `ac4f17c` | 2025-12-12 | CharmsTok | 回滚 `.gitignore` 修改
26. `b2b916b` | 2025-12-12 | CharmsTok | 重新应用 `.gitignore` 修改
27. `1fd39ff` | 2025-12-12 | CharmsTok | 再次回滚 `.gitignore` 修改
28. `8b5ea98` | 2025-12-12 | CharmsTok | 回滚 `.github` 删除
29. `917817b` | 2025-12-12 | CharmsTok | 再次应用 `.gitignore` 修改
30. `5ed5e01` | 2025-12-12 | CharmsTok | 恢复
31. `89465fc` | 2025-12-15 | CharmsTok | 新增 `AGING` 调度策略
32. `259fda1` | 2025-12-15 | CharmsTok | 删除 `.github`
33. `6056a3a` | 2025-12-15 | CharmsTok | 新增 `SRTF` 调度策略
34. `c83e2b3` | 2025-12-15 | CharmsTok | 初始化 `example` 目录
35. `541fa89` | 2025-12-16 | CharmsTok | 减少不必要的 import
36. `b1f81a8` | 2025-12-16 | CharmsTok | 初版 chunk 阈值选择逻辑与脚本
37. `63f4416` | 2025-12-16 | CharmsTok | 修正文案
38. `e4c91af` | 2025-12-16 | CharmsTok | 新增阈值选择脚本
39. `cb9fc73` | 2025-12-16 | CharmsTok | 新增阈值选择脚本
40. `fd6ba98` | 2025-12-17 | CharmsTok | 修改最大生成 token 数为 2048
41. `42f8c9c` | 2025-12-17 | CharmsTok | 调整模型参数以适配消费级显卡
42. `f757951` | 2025-12-17 | CharmsTok | 优化 `min_chunk_threshold` 及阈值脚本逻辑
43. `7960897` | 2025-12-18 | CharmsTok | 修复调度策略问题并清理 `AGING` 偏置项
44. `7f96db0` | 2025-12-19 | CharmsTok | modify
45. `86b1394` | 2025-12-19 | CharmsTok | README 新增 `prompt_weight` 说明
46. `a8b2e3d` | 2025-12-19 | CharmsTok | 调整文件名为 `threshold`
47. `a84ebb7` | 2025-12-19 | CharmsTok | 修改 README
48. `acd64bd` | 2025-12-19 | CharmsTok | modify readme
49. `e21ef34` | 2025-12-19 | CharmsTok | 调整目录名
50. `6e7cb70` | 2025-12-19 | CharmsTok | 增加 batch 特征统计并写入 CSV
51. `bb83b44` | 2025-12-22 | CharmsTok | 为 CSV 文件名添加时间戳
52. `9b338f0` | 2025-12-22 | CharmsTok | 回滚 CSV 时间戳提交
53. `a3eeeb6` | 2025-12-22 | CharmsTok | 再次应用 CSV 时间戳提交
54. `d34b538` | 2025-12-22 | CharmsTok | 增加新的离线测试样例并导出 batch 指标
55. `6544ae5` | 2025-12-23 | CharmsTok | 新增请求到达时间模拟方法
56. `e169911` | 2025-12-24 | CharmsTok | 新增平滑与聚集两种请求到达时间模拟
57. `f398d47` | 2025-12-24 | CharmsTok | 修复问题
58. `6d016a7` | 2025-12-24 | CharmsTok | 优化 `AGING` 参数选择逻辑
59. `d13922b` | 2025-12-25 | CharmsTok | 优化 CSV 存储位置并放开筛选条件
60. `262adfd` | 2025-12-25 | CharmsTok | CSV 增加 decode 历史长度列
61. `82f7184` | 2025-12-26 | CharmsTok | 删除旧实验脚本 `offline_select_status_csv.py`
62. `40bca37` | 2025-12-31 | lhx-666-cool | 备份旧版 `qwen.py`
63. `e89fdff` | 2025-12-31 | lhx-666-cool | 重建 `qwen.py` 以修复多卡问题
64. `d7eae98` | 2025-12-31 | lhx-666-cool | 修改 `.gitignore`
65. `d03d000` | 2025-12-31 | lhx-666-cool | 让 `AGING` 支持多卡场景
66. `717e99b` | 2025-12-31 | lhx-666-cool | 将离线示例的 `max_num_seqs` 提高到 32
67. `aa8aa7d` | 2025-12-31 | lhx-666-cool | 调整 `policy.py` 中的 `time_weight`
68. `fb26de7` | 2026-01-06 | lhx-666-cool | 再次更新 `policy.py`

## 7. 一句话结论

从 `26c49f13` 到 `fb26de78`，这条分支的核心成果是把项目改造成了一个以 `Opt-Sarathi + Qwen/Qwen3 + 离线阈值调参` 为中心的研究实验版本，并在区间末尾补上了 `Qwen` 多卡运行和 `AGING` 多卡生效这两个关键能力。
