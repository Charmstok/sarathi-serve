import csv
import statistics
import sys
from pathlib import Path

BASE_OUTPUT_DIR = Path("./offline_inference_output")


def latest_run(suffix: str) -> Path:
    runs = sorted(BASE_OUTPUT_DIR.glob(f"*-active_prefill_control-{suffix}"))
    if not runs:
        raise FileNotFoundError(f"未找到后缀为 {suffix} 的实验输出")
    return runs[-1]



def read_sequence_metrics(run_dir: Path) -> dict:
    path = run_dir / "replica_0" / "sequence_metrics.csv"
    request_e2e = []
    prefill_e2e = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            request_e2e.append(float(row["request_e2e_time"]))
            prefill_e2e.append(float(row["prefill_e2e_time"]))
    return {
        "request_e2e_mean": statistics.fmean(request_e2e),
        "prefill_e2e_mean": statistics.fmean(prefill_e2e),
        "request_e2e_p90": statistics.quantiles(request_e2e, n=10)[-1],
        "prefill_e2e_p90": statistics.quantiles(prefill_e2e, n=10)[-1],
    }



def read_active_prefill_stats(run_dir: Path) -> dict:
    path = run_dir / "replica_0" / "active_prefill_control_stats_rank0.csv"
    scheduled_prefill_seq_count = []
    avg_prefill_chunk = []
    deferred_prefill_seq_count = 0
    blocked_by_cap = 0
    blocked_by_min_chunk = 0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheduled_prefill_seq_count.append(int(float(row["scheduled_prefill_seq_count"])))
            avg_prefill_chunk.append(float(row["avg_prefill_chunk"]))
            deferred_prefill_seq_count += int(float(row["deferred_prefill_seq_count"]))
            blocked_by_cap += int(float(row["waiting_prefill_blocked_by_cap"]))
            blocked_by_min_chunk += int(float(row["waiting_prefill_blocked_by_min_chunk"]))
    return {
        "scheduled_prefill_seq_mean": statistics.fmean(scheduled_prefill_seq_count),
        "avg_prefill_chunk_mean": statistics.fmean(avg_prefill_chunk),
        "deferred_prefill_seq_total": deferred_prefill_seq_count,
        "blocked_by_cap_total": blocked_by_cap,
        "blocked_by_min_chunk_total": blocked_by_min_chunk,
    }



def summarize(run_dir: Path) -> dict:
    metrics = read_sequence_metrics(run_dir)
    active_stats = read_active_prefill_stats(run_dir)
    return {**metrics, **active_stats}



def print_summary(label: str, summary: dict) -> None:
    print(f"[{label}]")
    for key, value in summary.items():
        print(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}")
    print()



def main() -> None:
    if len(sys.argv) == 3:
        off_dir = Path(sys.argv[1])
        on_dir = Path(sys.argv[2])
    else:
        off_dir = latest_run("off")
        on_dir = latest_run("on")

    off_summary = summarize(off_dir)
    on_summary = summarize(on_dir)

    print(f"OFF_DIR={off_dir.resolve()}")
    print(f"ON_DIR={on_dir.resolve()}")
    print()
    print_summary("off", off_summary)
    print_summary("on", on_summary)

    print("[delta:on-off]")
    for key in off_summary:
        delta = on_summary[key] - off_summary[key]
        print(f"{key}={delta:.6f}" if isinstance(delta, float) else f"{key}={delta}")


if __name__ == "__main__":
    main()
