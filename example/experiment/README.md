# 实验

## 1. heterogeneous_prompt

异构场景实验：

- prompt 长度长短差异很大
- prefill / decode 混合变化明显

目的：

验证“负载越异构，时间预算越有优势”。

当前目录下有两个实验脚本：

- `example/experiment/heterogeneous_prompt/chunk_size.py`
  - 使用 `SarathiSchedulerConfig`
  - 固定 `chunk_size`，观察异构 prompt 负载下的分批效果
- `example/experiment/heterogeneous_prompt/target_time.py`
  - 使用 `OptSarathiSchedulerConfig`
  - 固定 `target_time`，观察时间预算调度在异构负载下的收益

### 负载构造方式

两个脚本都不再直接顺序读取普通 prompt 列表，而是通过
`sarathi.utils.prompt_utils.build_heterogeneous_prompt_dataset(...)`
构造“两段式极端异构 prompt”：

- 从 `dataset/ShareGPT_V3_unfiltered_cleaned_split.json` 中提取首个 human prompt
- 只保留两种 prompt：
  - short prompt: `10 ~ 30` tokens
  - long prompt: `500 ~ 512` tokens
- short / long 的比例可配置，默认 `1:1`
- 请求顺序会在满足 short / long 比例后随机打散，但同一 `seed` 下可复现
- 为每条 prompt 单独分配 `num_decode_tokens`
- short prompt 默认走 `decode_heavy`
- long prompt 默认走 `prefill_heavy`
- 每条请求会用自己的 `SamplingParams(max_tokens=record["num_decode_tokens"])` 提交给引擎
- 实验脚本默认设置 `SARATHI_SAMPLING_BACKEND=torch`，避免 `flashinfer` 采样在高异构负载下出现不稳定 warning

这样得到的 workload 同时具备：

- prefill token 数差异大
- decode token 数差异大
- `prefill / decode` 比例变化明显
- 同一组参数和 `seed` 下可复现

### 默认实验参数

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

### 输出文件

每次运行会在对应 `output_dir` 下额外保存：

- `config.json`
  - 包含调度参数、异构 prompt 参数和本次 workload 摘要
- `heterogeneous_prompts.json`
  - 位于对应的 `replica_0/` 目录下
  - 包含本次实验实际使用的 prompt 文本、`prompt_type`、`num_prefill_tokens`、`num_decode_tokens`、`pd_ratio`、`decode_regime`

### 运行方式

在项目根目录执行：

```bash
python example/experiment/heterogeneous_prompt/chunk_size.py
python example/experiment/heterogeneous_prompt/target_time.py
```

### 调参建议

如果你想进一步强化异构性，优先调这几个参数：

- 拉大 short / long 两段区间之间的间隔
- 调整 `SHORT_LONG_RATIO`
- 提高 long prompt 下界
- 提高 `model_config.max_model_len`

如果你想保证实验更严格可复现，不要改：

- `DATA_SOURCE`
- `HETEROGENEOUS_PROMPT_SEED`
- tokenizer 对应模型
