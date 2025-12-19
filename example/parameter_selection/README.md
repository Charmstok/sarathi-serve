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
- 每次运行会读取 `offline_inference_output` 下最新生成的时间戳目录，从其中的 `replica_0/plots` 复制若干 CSV（用于后续选参分析）：
  - `request_execution_time.csv`
  - `request_execution_time_normalized.csv`
  - `prefill_e2e_time.csv`
  - `prefill_time_execution_plus_preemption_normalized.csv`
  - `decode_time_execution_plus_preemption_normalized.csv`
- 结果按阈值分目录存放在 `example/parameter_selection/threshold_output/<脚本启动时间>/<threshold>/`，原始离线输出目录不会被删除。

CSV 的含义（每行包含 `Request Id`、指标值、以及该指标的经验分布 `cdf`）：
- `request_execution_time.csv`：请求在 GPU 上真正“执行”的总时长（跨所有尝试累计），不包含在 replica 上分配了资源但因 pipeline bubble / 调度抢占 / 重启间隔等导致的非执行时间。
- `request_execution_time_normalized.csv`：`request_execution_time` 按总 token 数归一化后的单位 token 平均纯执行耗时（整体效率代理）。
- `prefill_e2e_time.csv`：首 token 延迟（TTFT），即从请求到达系统到首个输出 token 产生的端到端耗时。
- `prefill_time_execution_plus_preemption_normalized.csv`：prefill 阶段总耗时（含执行+抢占/等待，但不含初始排队）按 prefill token 数归一化后的单位 token 平均成本（prefill 效率代理）。
- `decode_time_execution_plus_preemption_normalized.csv`：decode 阶段总耗时按 decode token 数归一化后的平均值，近似反映“每个 decode token 的平均间隔/耗时”（包含执行 + 抢占/等待等），计算形式可理解为 `(request_completion_time - prefill_completion_time) / num_decode_tokens`。

可根据需要调整脚本中的：
- `MAX_THRESHOLD`：阈值最大值（包含该值）。
- `RUN_TS` 逻辑：输出目录的时间戳格式。

为避免偶然误差，可多次执行上述 bash 脚本。

### 2. 选择阈值

`select_threshold.py` 会读取 `threshold_output/<时间>/<threshold>/` 下的 CSV，并对每次运行先按基准阈值（默认 0）做归一化，再对多次运行聚合（默认取均值），最后给出推荐阈值：
- 指标摘要：`request_execution_time` 的 P95、`prefill_e2e_time` 的 P95、`prefill_time_execution_plus_preemption_normalized` 的均值、`request_execution_time_normalized` 的均值、`decode_time_execution_plus_preemption_normalized` 的均值。
- 归一化：对每个 run 分别计算 ratio = metric(threshold) / metric(baseline)，再跨 run 聚合（mean/median/p90/max），减小偶然波动影响。
- 选择逻辑：计算综合评分 `score`，选择 `score` 最小的阈值（等价于 `min_score`）。

### select_threshold.py 参数说明

直接运行 `select_threshold.py` 并传参即可（所有参数都是可选的）：

- `--root <path>`：阈值输出根目录（默认 `example/parameter_selection/threshold_output`），其下应包含多个时间戳目录，每个时间戳目录内包含多个 `<threshold>/` 子目录及三个 CSV。
- `--baseline <int>`：归一化使用的基准阈值（默认 `0`）。若某个 run 没有该阈值目录，会自动回退为该 run 的最小阈值作为 baseline。
- `--w_request <float>`：综合评分里 `request_ratio` 的权重（默认 `0.35`），越大越偏向优化 request 尾延迟。
- `--w_prefill <float>`：综合评分里 `prefill_ratio` 的权重（默认 `0.35`），越大越偏向优化 TTFT 尾延迟。
- `--w_prefill_exec_norm <float>`：综合评分里 `prefill_exec_norm_ratio` 的权重（默认 `0.2`），用于强调“prefill 单位 token 成本”（效率代理）。
- `--w_request_exec_norm <float>`：综合评分里 `request_exec_norm_ratio` 的权重（默认 `0.05`），用于强调“整体单位 token 纯执行成本”（效率代理）。
- `--w_decode <float>`：综合评分里 `decode_ratio` 的权重（默认 `0.05`），用于约束 decode 不要明显变差。
- `--run_agg <choice>`：跨多个 run 聚合 ratio 的方式（默认 `mean`）。`median` 更稳健（抗偶然异常），`p90/max` 更保守（倾向“坏情况下也不回归太多”）。

示例：

```shell
# 默认
python example/parameter_selection/select_threshold.py
```

```shell
# 通过权重更强调 prefill 效率代理（prefill_exec_norm_ratio）
python example/parameter_selection/select_threshold.py --w_prefill_exec_norm 0.35 --w_request 0.25 --w_prefill 0.25
```

```shell
# 多次跑基准后，用更稳健/更保守的跨 run 聚合方式选参
python example/parameter_selection/select_threshold.py --run_agg median
python example/parameter_selection/select_threshold.py --run_agg p90
```

---

## 参数 2：AGING策略的 `prompt_weight`

要把 AGING 的参数（本质是 time_weight 和 prompt_weight）选得“更对”，核心是把两个量统一到同一尺度，并且明确**你想优化的目标**（TTFT 还是整体 E2E、吞吐还是公平）。

优先级设定公式：`priority = time_weight * wait_seconds + prompt_weight * remaining_prompt_tokens`

最简单、也最“物理一致”的做法是把 `prompt_weight` 设成 “**每个 prefill token 需要的秒数（取负号）**”，每个 prefill token 需要的秒数可以从基准测试结果（位于 ./offline_inference_output/{time}/replica_0/prefill_time_execution_plus_preemption_normalized.csv）。

你可以选择使用离线测试时 `prefill_time_execution_plus_preemption_normalized` 的 **平均数 / 中位数** 等数值作为`prompt_weight`。

当然，你也可以使用默认的数值 `0.01`。

---