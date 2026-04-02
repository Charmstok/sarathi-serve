# active_prefill_control

该实验对比两种设置：

- `baseline.py`：关闭 active prefill 控制。
- `enabled.py`：开启 active prefill 并发上限与最小有效 chunk 控制。

统一 workload 特征：

- `max_num_seqs=32`
- `chunk_size=512`
- `target_time=100`
- clustered arrival
- short prompt 主导，且 short prompt 对应 decode-heavy workload

新增输出文件：

- `active_prefill_control_stats_rank0.csv`
  - `active_prefill_seq_cap`
  - `active_prefill_seq_count`
  - `deferred_prefill_seq_count`
  - `waiting_prefill_blocked_by_cap`
  - `waiting_prefill_blocked_by_min_chunk`
  - `scheduled_prefill_seq_count`
  - `scheduled_prefill_tokens`
  - `avg_prefill_chunk`

运行方式：

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/baseline.py
./env/bin/python example/experiment/active_prefill_control/enabled.py
./env/bin/python example/experiment/active_prefill_control/compare_runs.py
```

或者直接顺序运行：

```bash
cd /home/ta/project/sarathi-serve
./env/bin/python example/experiment/active_prefill_control/run_pair.py
```
