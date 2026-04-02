# 实验

本目录用于存放围绕调度策略开展的可复现实验脚本。当前主要包含两类实验：

- `heterogeneous_prompt/`
  - 用于验证在强异构 prompt 负载下，固定 `chunk_size` 与固定 `target_time` 两类调度方式的差异。
- `active_prefill_control/`
  - 用于验证 `Active Prefill Control (APC)` 这类 unfinished prefill 活跃度控制策略是否能够改变 batch 结构，并进一步影响 `prefill_e2e_time` 与 `request_e2e_time`。

所有实验脚本默认都在项目根目录下运行，并将结果输出到 `./offline_inference_output/` 下带时间戳的目录中。

---

## 1. 目录概览

当前实验目录结构如下：

- `example/experiment/README.md`
  - 本说明文档。
- `example/experiment/heterogeneous_prompt/`
  - 异构 prompt 负载实验。
- `example/experiment/active_prefill_control/`
  - APC 开关对比实验。

如果后续继续新增实验，建议保持每个实验子目录内至少包含：

- 一个公共参数文件或公共工具文件
- 一个或多个可直接运行的实验脚本
- 一个局部 README，用于记录该实验的特殊参数与输出说明

---

## 2. 通用运行约定

### 2.1 运行位置

所有脚本都默认从项目根目录执行：

```bash
cd /home/ta/project/sarathi-serve
```

### 2.2 Python 环境

推荐统一使用项目虚拟环境：

```bash
./env/bin/python ...
```

### 2.3 输出位置

每次运行都会在 `./offline_inference_output/` 下生成独立目录，目录名通常包含：

- 时间戳
- 实验名
- 实验标签，如 `on/off`

典型输出内容包括：

- `config.json`
  - 保存本次实验的关键配置、脚本路径和 workload 摘要。
- `replica_0/heterogeneous_prompts.json`
  - 保存本次实验实际使用的异构 prompt 数据集。
- `replica_0/select_stats_rank0.csv`
  - 保存逐 batch 的调度选择统计。
- `replica_0/sequence_metrics.csv`
  - 保存请求级指标，如 `prefill_e2e_time` 与 `request_e2e_time`。

若某个实验额外记录了新统计文件，会在该实验自己的 README 或下文对应小节中说明。

### 2.4 可复现性

当前实验脚本普遍采用如下可复现设计：

- 固定数据源：`dataset/ShareGPT_V3_unfiltered_cleaned_split.json`
- 固定异构 prompt 构造 `seed`
- 固定 prompt 长度区间和 short/long 比例
- 固定到达模式，如 clustered arrival

因此，只要模型、tokenizer 和数据源不变，同一组脚本参数通常可以复现实验 workload。

---

## 3. heterogeneous_prompt

### 3.1 实验目的

该实验用于构造高度异构的 prompt 负载，以验证在 `prefill/decode` 混合程度变化明显的场景下，时间预算调度是否相较固定 `chunk_size` 更有优势。

该实验的核心关注点是：

- prompt 长度分布是否足够异构
- `prefill/decode` 比例是否显著波动
- 固定 `target_time` 是否比固定 `chunk_size` 更适应这种异构负载

### 3.2 脚本说明

当前包含两个脚本：

- `example/experiment/heterogeneous_prompt/chunk_size.py`
  - 使用 `SarathiSchedulerConfig`
  - 固定 `chunk_size`，观察异构 prompt 负载下的分批效果
- `example/experiment/heterogeneous_prompt/target_time.py`
  - 使用 `OptSarathiSchedulerConfig`
  - 固定 `target_time`，观察时间预算调度在异构负载下的收益

### 3.3 负载构造方式

两个脚本都不再直接顺序读取普通 prompt 列表，而是通过
`sarathi.utils.prompt_utils.build_heterogeneous_prompt_dataset(...)`
构造“两段式极端异构 prompt”：

- 从 `dataset/ShareGPT_V3_unfiltered_cleaned_split.json` 中提取首个 human prompt
- 只保留两种 prompt：
  - short prompt: `10 ~ 30` tokens
  - long prompt: `500 ~ 512` tokens
- `short / long` 比例可配置，默认 `1:1`
- 请求顺序会在满足 short/long 比例后随机打散，但同一 `seed` 下可复现
- 为每条 prompt 单独分配 `num_decode_tokens`
- short prompt 默认走 `decode_heavy`
- long prompt 默认走 `prefill_heavy`
- 每条请求会用自己的 `SamplingParams(max_tokens=record["num_decode_tokens"])` 提交给引擎
- 脚本默认设置 `SARATHI_SAMPLING_BACKEND=torch`，避免 `flashinfer` 采样在高异构负载下出现不稳定 warning

因此，该 workload 同时具备：

- prefill token 数差异大
- decode token 数差异大
- `prefill/decode` 比例变化明显
- 同一组参数和 `seed` 下可复现

### 3.4 默认实验参数

两个脚本当前使用同一组异构 prompt 参数：

- `PROMPTS_NUMBER = 200`
- `model_config.max_model_len = 1024`
- `HETEROGENEOUS_PROMPT_SEED = 42`
- `SHORT_PROMPT_MIN_TOKENS = 10`
- `SHORT_PROMPT_MAX_TOKENS = 30`
- `LONG_PROMPT_MIN_TOKENS = 500`
- `LONG_PROMPT_MAX_TOKENS = 512`
- `SHORT_LONG_RATIO = (1, 1)`
- `MIN_DECODE_TOKENS = 16`

### 3.5 输出文件

每次运行会在对应 `output_dir` 下额外保存：

- `config.json`
  - 包含调度参数、异构 prompt 参数和本次 workload 摘要
- `replica_0/heterogeneous_prompts.json`
  - 包含本次实验实际使用的 prompt 文本、`prompt_type`、`num_prefill_tokens`、`num_decode_tokens`、`pd_ratio`、`decode_regime`

### 3.6 运行方式

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/heterogeneous_prompt/chunk_size.py
./env/bin/python example/experiment/heterogeneous_prompt/target_time.py
```

### 3.7 调参建议

如果你想进一步强化异构性，优先调以下参数：

- 拉大 short/long 两段区间之间的间隔
- 调整 `SHORT_LONG_RATIO`
- 提高 long prompt 下界
- 提高 `model_config.max_model_len`

如果你想保持实验严格可复现，尽量不要改：

- `DATA_SOURCE`
- `HETEROGENEOUS_PROMPT_SEED`
- tokenizer 对应模型

---

## 4. active_prefill_control

### 4.1 实验目的

该实验用于对比在相同 workload 条件下，关闭与开启 `Active Prefill Control (APC)` 时的行为差异。这里的 APC 关注 unfinished prefill 的活跃度控制，而不是静态预留固定槽位。

该实验主要回答三类问题：

- 开启 APC 后，unfinished prefill 是否真的发生了“受控活跃”而非完全不触发
- 开启 APC 后，batch 结构是否发生变化，例如平均 prefill chunk 是否变大、decode-only batch 是否减少
- 这些结构变化最终是否会传导到 `prefill_e2e_time` 与 `request_e2e_time`

### 4.2 脚本说明

当前目录包含以下脚本：

- `example/experiment/active_prefill_control/common.py`
  - 公共参数、workload 构造、实验执行入口
- `example/experiment/active_prefill_control/baseline.py`
  - 关闭 APC，作为对照组
- `example/experiment/active_prefill_control/enabled.py`
  - 开启 APC，作为实验组
- `example/experiment/active_prefill_control/run_pair.py`
  - 依次运行 `baseline.py` 和 `enabled.py`，然后自动调用对比脚本
- `example/experiment/active_prefill_control/compare_runs.py`
  - 对两次运行结果进行汇总，输出均值、P90 与 APC 行为统计
- `example/experiment/active_prefill_control/README.md`
  - 该实验目录自己的简版说明

### 4.3 当前 workload 特征

`common.py` 当前默认构造的是一个更偏向 `decode-heavy` 的 clustered workload，用于放大 unfinished prefill 与 decode 的竞争关系。关键参数包括：

- `PROMPTS_NUMBER = 240`
- `TARGET_TIME = 100`
- `ARRIVAL_INTERVAL_S = 0.025`
- `CLUSTER_START_PCT = 0.10`
- `CLUSTER_END_PCT = 0.30`
- `SHORT_PROMPT_MIN_TOKENS = 30`
- `SHORT_PROMPT_MAX_TOKENS = 50`
- `LONG_PROMPT_MIN_TOKENS = 200`
- `LONG_PROMPT_MAX_TOKENS = 220`
- `SHORT_LONG_RATIO = (49, 1)`
- `MIN_DECODE_TOKENS = 4`
- `MAX_MODEL_LEN = 256`
- `CHUNK_SIZE = 512`
- `MAX_NUM_SEQS = 32`
- `GPU_MEMORY_UTILIZATION = 0.65`

APC 相关默认参数为：

- `MAX_ACTIVE_PREFILL_SEQS = 6`
- `MIN_ACTIVE_PREFILL_CHUNK_SIZE = 16`

### 4.4 输出文件

除了通用输出文件之外，该实验还会额外生成：

- `replica_0/active_prefill_control_stats_rank0.csv`
  - APC 的逐 batch 行为统计文件
  - 主要字段包括：
    - `active_prefill_seq_cap`
    - `active_prefill_seq_count`
    - `deferred_prefill_seq_count`
    - `waiting_prefill_blocked_by_cap`
    - `waiting_prefill_blocked_by_min_chunk`
    - `scheduled_prefill_seq_count`
    - `scheduled_prefill_tokens`
    - `avg_prefill_chunk`

`compare_runs.py` 当前会读取：

- `replica_0/sequence_metrics.csv`
- `replica_0/active_prefill_control_stats_rank0.csv`

并输出：

- `request_e2e_mean`
- `prefill_e2e_mean`
- `request_e2e_p90`
- `prefill_e2e_p90`
- `scheduled_prefill_seq_mean`
- `avg_prefill_chunk_mean`
- `deferred_prefill_seq_total`
- `blocked_by_cap_total`
- `blocked_by_min_chunk_total`
- `delta:on-off`

### 4.5 运行方式

单独运行：

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/baseline.py
./env/bin/python example/experiment/active_prefill_control/enabled.py
```

使用自动对比脚本：

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/run_pair.py
```

如果你已经有一组 `off/on` 输出目录，也可以手动指定给对比脚本：

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/compare_runs.py \
  /path/to/off_dir \
  /path/to/on_dir
```

### 4.6 建议观察指标

当你分析 APC 实验时，建议优先看以下几组指标：

- 请求侧：
  - `prefill_e2e_time`
  - `request_e2e_time`
- batch 结构侧：
  - `scheduled_prefill_seq_count`
  - `avg_prefill_chunk`
  - `decode-only batch` 比例
- APC 行为侧：
  - `active_prefill_seq_count`
  - `deferred_prefill_seq_count`
  - `waiting_prefill_blocked_by_cap`
  - `waiting_prefill_blocked_by_min_chunk`

如果 `active_prefill_control_stats_rank0.csv` 中长期看不到 `active_prefill_seq_count > 0`，通常意味着：

- 当前 workload 还不够容易触发 APC
- `C_max` 太大，APC 退化为接近无约束放行
- `L_min` 太小，APC 缺乏最小有效推进约束
- 或者相反，`L_min` 太大，导致 prefill 很难进入 active 集合

---

## 5. 推荐使用顺序

如果你是第一次使用这些实验脚本，建议按以下顺序：

1. 先运行 `heterogeneous_prompt/target_time.py`，确认时间预算实验链路可正常工作。
2. 再运行 `active_prefill_control/baseline.py` 与 `enabled.py`，确认 APC 开关实验能正常产出 `sequence_metrics.csv` 与 `active_prefill_control_stats_rank0.csv`。
3. 最后使用 `run_pair.py` 或 `compare_runs.py` 做成对对比。

---

## 6. 后续扩展建议

如果后续继续在 `example/experiment/` 下新增实验，建议统一遵循以下约定：

- 每个实验子目录至少包含一个局部 README
- 统一将公共参数放入 `common.py` 或等价文件中
- 统一通过 `config.json` 保存运行配置与 workload 摘要
- 若新增调度行为统计，不要修改现有 `select_stats_rank0.csv` 语义，优先新增独立统计文件
- 若提供 `on/off` 对比，建议同时提供一个自动汇总脚本
