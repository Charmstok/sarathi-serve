# 调整参数

## 参数 1：AGING策略的 `prompt_weight`

要把 AGING 的参数（time_weight 和 prompt_weight）选得“更对”，核心是把两个量统一到同一尺度，并且明确**你想优化的目标**（TTFT 还是整体 E2E、吞吐还是公平）。

优先级设定公式：`priority = time_weight * wait_seconds + prompt_weight * remaining_prompt_tokens`

最简单、也最“物理一致”的做法是
- `time_weight` 设成 “每秒可以处理的 prefill token数”。这个值可以由 `1` 除以 **每个 prefill token 需要的秒数**”得到，每个 prefill token 需要的秒数可以从基准测试结果（位于 ./offline_inference_output/{time}/replica_0/prefill_time_execution_plus_preemption_normalized.csv），选择 **P80 / 平均数**。

- `prompt_weight` 设成 `-1.0`

> 如果是为了吞吐量优先，可以减小 `time_weight`，让系统更偏向短请求 。
>
> 例如，默认数值 `125 * 0.3`, `-1.0` 就更偏向于短请求。

---

# 训练“时间预算”模型

本项目的 “时间预算” 调度依赖 `sarathi/time_balance/predict_time.py` 训练得到的 `TimePredictor`（MLP 回归器），用于在调度阶段预测某个 batch 的执行耗时（ms）。

训练流程分两步：先离线收集 `select_stats_rank*.csv`，再训练并保存模型。

## 1. 离线收集训练数据（select stats CSV）

先运行离线脚本，它会用 `SarathiScheduler` 跑一批请求，并在模型执行阶段写出 `select_stats_rank0.csv`：

```sh
python example/time_balance/offline_select_status_csv.py
```

说明：
- 脚本输出目录在 `offline_inference_output/<时间戳>/replica_0/`。
- 关键产物是 `select_stats_rank0.csv`（包含 `decode_tokens/prefill_tokens/.../latency_ms` 等特征与标签）。
- 若你改了脚本里的 `chunk_size/max_num_seqs/max_model_len` 等参数，建议在训练配置里保持一致（尤其是 `chunk_size`）。

## 2. 配置训练路径与分桶参数

打开 `sarathi/time_balance/config.py`，按你刚生成的数据修改：
- `CSV_PATH`：指向你最新目录下的 `select_stats_rank0.csv`
- `MODEL_CACHE_PATH`：模型保存路径（默认在 `sarathi/time_balance/time_predictor_mlp_v6.pt`）
- `BUCKET_SPLIT_CONFIG.chunk_size`：建议与数据采集时的 scheduler `chunk_size` 一致（默认 256）

## 3. 训练并保存模型

执行：

```sh
python sarathi/time_balance/predict_time.py
```

训练完成后会输出 train/val/test 的 MAE，并将模型写入 `MODEL_CACHE_PATH`。

## 4. 使用说明（OptSarathiScheduler）

`OptSarathiScheduler` 会在初始化时从 `sarathi/time_balance/config.py` 的 `MODEL_CACHE_PATH` 加载模型；如果文件不存在会直接 `assert` 报错（确保线上不会静默退化）。

可用下面脚本快速验证模型能否正确加载并预测：

```sh
python sarathi/time_balance/load_model.py
```
