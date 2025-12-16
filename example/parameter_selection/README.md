# 调整参数

## 参数 1: 最大填充阈值(max_chunk_threshold)

调度器的批处理填充做法可能引入额外的调度与计算开销且收益有限。因此，拟在剩余预算较小时引入更优的填充策略。

在预算过小（低于阈值）时选择本轮不再填充，以减少无效或低效的计算。

### 1. 获取基准测试结果

```shell
bash example/parameter_selection/offline_get_threshold.sh
```

脚本说明：
- 顺序运行 `offline_get_threshold.py`，将 `--min_chunk_threshold` 从 0 递增到脚本内的 `MAX_THRESHOLD`（默认 10，可自行修改）。
- 每次运行会读取 `offline_inference_output` 下最新生成的时间戳目录，从其中的 `replica_0/plots` 复制三个 CSV（`request_execution_time.csv`、`prefill_e2e_time.csv`、`decode_time_execution_plus_preemption_normalized.csv`）。
- 结果按阈值分目录存放在 `example/parameter_selection/threshold_output/<脚本启动时间>/<threshold>/`，原始离线输出目录不会被删除。

可根据需要调整脚本中的：
- `MAX_THRESHOLD`：阈值最大值（包含该值）。
- `RUN_TS` 逻辑：输出目录的时间戳格式。

为避免偶然误差，可多次执行上述 bash 脚本。

### 2. 选择最大阈值

```shell
python example/parameter_selection/select_threshold.py
```

