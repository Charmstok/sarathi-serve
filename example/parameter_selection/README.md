# 调整参数

## 参数 1: 最小分块阈值(min_chunk_threshold)

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

三个 CSV 的含义（每行包含 `Request Id`、指标值、以及该指标的经验分布 `cdf`）：
- `request_execution_time.csv`：请求在 GPU 上真正“执行”的总时长（跨所有尝试累计），不包含在 replica 上分配了资源但因 pipeline bubble / 调度抢占 / 重启间隔等导致的非执行时间。
- `prefill_e2e_time.csv`：首 token 延迟（TTFT），即从请求到达系统到首个输出 token 产生的端到端耗时。
- `decode_time_execution_plus_preemption_normalized.csv`：decode 阶段总耗时按 decode token 数归一化后的平均值，近似反映“每个 decode token 的平均间隔/耗时”（包含执行 + 抢占/等待等），计算形式可理解为 `(request_completion_time - prefill_completion_time) / num_decode_tokens`。

可根据需要调整脚本中的：
- `MAX_THRESHOLD`：阈值最大值（包含该值）。
- `RUN_TS` 逻辑：输出目录的时间戳格式。

为避免偶然误差，可多次执行上述 bash 脚本。

### 2. 选择最大阈值

`select_threshold.py` 会读取 `threshold_output/<时间>/<threshold>/` 下的三个 CSV，并对每次运行先按基准阈值（默认 0）做归一化，再对多次运行取平均，最后按策略给出推荐阈值：
- 指标摘要：`request_execution_time` 的 P95、`prefill_e2e_time` 的 P95、`decode_time_execution_plus_preemption_normalized` 的均值。
- 归一化：对每个 run 分别计算 ratio = metric(threshold) / metric(baseline)，再跨 run 求平均，减小偶然波动影响。
- 策略（默认）：在 `request_ratio` 与 `prefill_ratio` 回归不超过阈值（默认 +2%）的前提下，选择最大的 `min_chunk_threshold`。

### select_threshold.py 参数说明

直接运行 `select_threshold.py` 并传参即可（所有参数都是可选的）：

- `--root <path>`：阈值输出根目录（默认 `example/parameter_selection/threshold_output`），其下应包含多个时间戳目录，每个时间戳目录内包含多个 `<threshold>/` 子目录及三个 CSV。
- `--baseline <int>`：归一化使用的基准阈值（默认 `0`）。若某个 run 没有该阈值目录，会自动回退为该 run 的最小阈值作为 baseline。
- `--w_request <float>`：综合评分里 `request_ratio` 的权重（默认 `0.6`），越大越偏向优化 request 尾延迟。
- `--w_prefill <float>`：综合评分里 `prefill_ratio` 的权重（默认 `0.2`），越大越偏向优化 prefill 尾延迟。
- `--w_decode <float>`：综合评分里 `decode_ratio` 的权重（默认 `0.2`），越大越偏向优化 decode 平均开销（含抢占归一化）。
- `--strategy <choice>`：阈值选择策略（默认 `max_threshold_within_regression`）。
  - `max_threshold_within_regression`：在 `request_ratio` 与 `prefill_ratio` 回归不超过阈值（见下两个参数）的前提下，取 **最大的** `min_chunk_threshold`（更激进地拒绝“微小分块”）。
  - `min_score`：直接选择综合评分 `score = w_request*request_ratio + w_prefill*prefill_ratio + w_decode*decode_ratio` 最小的阈值。
- `--max_request_regression <float>`：允许的 `request_ratio` 最大回归比例（默认 `0.02`，即允许比 baseline 差 +2%）。
- `--max_prefill_regression <float>`：允许的 `prefill_ratio` 最大回归比例（默认 `0.02`，即允许比 baseline 差 +2%）。

示例：

```shell
# 默认
python example/parameter_selection/select_threshold.py
```

```shell
# 更保守：只允许 +1% 的 request/prefill 回归
python example/parameter_selection/select_threshold.py --max_request_regression 0.01 --max_prefill_regression 0.01
```

```shell
# 直接用最小 score 选阈值
python example/parameter_selection/select_threshold.py --strategy min_score
```
