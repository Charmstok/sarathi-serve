# 调整参数

## 参数 2：AGING策略的 `prompt_weight`

要把 AGING 的参数（time_weight 和 prompt_weight）选得“更对”，核心是把两个量统一到同一尺度，并且明确**你想优化的目标**（TTFT 还是整体 E2E、吞吐还是公平）。

优先级设定公式：`priority = time_weight * wait_seconds + prompt_weight * remaining_prompt_tokens`

最简单、也最“物理一致”的做法是
- `time_weight` 设成 “每秒可以处理的 prefill token数”。这个值可以由 `1` 除以 **每个 prefill token 需要的秒数**”得到，每个 prefill token 需要的秒数可以从基准测试结果（位于 ./offline_inference_output/{time}/replica_0/prefill_time_execution_plus_preemption_normalized.csv），选择 **P80 / 平均数**。

- `prompt_weight` 设成 `-1.0`

> 如果是为了吞吐量优先，可以减小 `time_weight`，让系统更偏向短请求 。
>
> 例如，默认数值 `125 * 0.3`, `-1.0` 就更偏向于短请求。

---
