import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple


"""
根据离线基准测试导出的 CSV，选择一个“最小分块阈值”(min_chunk_threshold)。

数据目录结构（由 offline_get_threshold.sh 生成）：
  threshold_output/<时间戳>/<threshold>/
    - request_execution_time.csv
    - prefill_e2e_time.csv
    - decode_time_execution_plus_preemption_normalized.csv

  允许多次重复跑 bash 脚本，把每次运行当成一个 run。
  本脚本会对每个 run 先做归一化，再跨 run 取平均，从而降低偶然误差。
"""


@dataclass(frozen=True)
class ThresholdMetrics:
    """某个阈值下的汇总指标（单个 threshold、单次 run）。"""

    request_p95_s: float
    prefill_p95_s: float
    decode_mean_s: float


def _quantile(values: List[float], q: float) -> float:
    """计算分位数（线性插值），用于 P95 之类的统计。"""

    if not values:
        raise ValueError("values is empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def _read_metric_values(csv_path: Path, metric_col: str) -> List[float]:
    """读取单个 CSV 的某一列（忽略空值），返回所有样本值。"""

    values: List[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric_col not in reader.fieldnames:
            raise ValueError(f"{csv_path} missing column {metric_col!r}")
        for row in reader:
            raw = row.get(metric_col)
            if raw is None or raw == "":
                continue
            values.append(float(raw))
    if not values:
        raise ValueError(f"{csv_path} has no values for column {metric_col!r}")
    return values


def _load_metrics_for_threshold(threshold_dir: Path) -> ThresholdMetrics:
    """
    从某个 threshold 目录加载并汇总三类指标：
    - request: 关注尾延迟，取 P95
    - prefill: 关注尾延迟，取 P95
    - decode: 关注整体 decode 开销，取均值
    """

    req_csv = threshold_dir / "request_execution_time.csv"
    prefill_csv = threshold_dir / "prefill_e2e_time.csv"
    decode_csv = threshold_dir / "decode_time_execution_plus_preemption_normalized.csv"

    request_vals = _read_metric_values(req_csv, "request_execution_time")
    prefill_vals = _read_metric_values(prefill_csv, "prefill_e2e_time")
    decode_vals = _read_metric_values(
        decode_csv, "decode_time_execution_plus_preemption_normalized"
    )

    return ThresholdMetrics(
        request_p95_s=_quantile(request_vals, 0.95),
        prefill_p95_s=_quantile(prefill_vals, 0.95),
        decode_mean_s=mean(decode_vals),
    )


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    """枚举 threshold_output 下的所有 run 目录（通常是时间戳目录）。"""

    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def _is_int_dir(p: Path) -> bool:
    try:
        int(p.name)
        return p.is_dir()
    except ValueError:
        return False


def load_all_runs(root: Path) -> Dict[str, Dict[int, ThresholdMetrics]]:
    """
    读取所有 run 的所有 threshold 指标。

    返回结构：
      { run_name(时间戳): { threshold(int): ThresholdMetrics } }
    """

    runs: Dict[str, Dict[int, ThresholdMetrics]] = {}
    for run_dir in _iter_run_dirs(root):
        thresholds: Dict[int, ThresholdMetrics] = {}
        for th_dir in run_dir.iterdir():
            if not _is_int_dir(th_dir):
                continue
            th = int(th_dir.name)
            try:
                thresholds[th] = _load_metrics_for_threshold(th_dir)
            except FileNotFoundError:
                continue
        if thresholds:
            runs[run_dir.name] = thresholds
    return runs


def _baseline_threshold(thresholds: Dict[int, ThresholdMetrics], preferred: int) -> int:
    """为一个 run 选择 baseline 阈值：优先用 preferred，否则用该 run 的最小阈值。"""

    if preferred in thresholds:
        return preferred
    return min(thresholds.keys())


def select_threshold(
    runs: Dict[str, Dict[int, ThresholdMetrics]],
    baseline_preferred: int = 0,
    w_request: float = 0.6,
    w_prefill: float = 0.2,
    w_decode: float = 0.2,
    strategy: str = "max_threshold_within_regression",
    max_request_regression: float = 0.02,
    max_prefill_regression: float = 0.02,
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    根据多次 run 的结果选择阈值。

    关键做法：每个 run 内先按 baseline 做归一化（ratio），再跨 run 求平均。
      ratio(th) = metric(th) / metric(baseline)

    这样可以减弱不同 run 的整体快慢差异（例如机器抖动/负载变化）。
    """

    # 每个 run 都用自己的 baseline 做归一化，降低 run-to-run 方差。
    per_threshold: Dict[int, Dict[str, List[float]]] = {}
    for _, thresholds in runs.items():
        base_th = _baseline_threshold(thresholds, baseline_preferred)
        base = thresholds[base_th]
        for th, m in thresholds.items():
            per_threshold.setdefault(th, {}).setdefault("request_ratio", []).append(
                m.request_p95_s / base.request_p95_s
            )
            per_threshold.setdefault(th, {}).setdefault("prefill_ratio", []).append(
                m.prefill_p95_s / base.prefill_p95_s
            )
            per_threshold.setdefault(th, {}).setdefault("decode_ratio", []).append(
                m.decode_mean_s / base.decode_mean_s
            )

    summary: Dict[int, Dict[str, float]] = {}
    for th, ratios in per_threshold.items():
        if not (
            ratios.get("request_ratio")
            and ratios.get("prefill_ratio")
            and ratios.get("decode_ratio")
        ):
            continue
        req = mean(ratios["request_ratio"])
        pre = mean(ratios["prefill_ratio"])
        dec = mean(ratios["decode_ratio"])
        score = w_request * req + w_prefill * pre + w_decode * dec
        summary[th] = {
            "runs": float(len(ratios["request_ratio"])),
            "request_ratio": req,
            "prefill_ratio": pre,
            "decode_ratio": dec,
            # score 越小越好；权重默认更偏向 request 尾延迟。
            "score": score,
        }

    if not summary:
        raise RuntimeError("No valid threshold data found under threshold_output/")

    if strategy == "min_score":
        best_th = min(summary.keys(), key=lambda th: (summary[th]["score"], th))
        return best_th, summary

    if strategy != "max_threshold_within_regression":
        raise ValueError(f"Unknown strategy: {strategy}")

    # 目标：在“性能回归可接受”的前提下，尽量选更大的阈值，减少微小分块带来的低效 kernel launch。
    candidates = [
        th
        for th, row in summary.items()
        if row["request_ratio"] <= 1.0 + max_request_regression
        and row["prefill_ratio"] <= 1.0 + max_prefill_regression
    ]
    if candidates:
        best_th = max(candidates)
    else:
        best_th = min(summary.keys(), key=lambda th: (summary[th]["score"], th))
    return best_th, summary


def main() -> None:
    """CLI 入口：打印每个阈值的汇总对比，并输出推荐阈值。"""

    parser = argparse.ArgumentParser(description="根据离线 CSV 选择最小分块阈值。")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("example/parameter_selection/threshold_output"),
        help="阈值输出根目录（包含多个时间戳子目录）。",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=0,
        help="用于归一化的基准阈值（缺失时自动回退到该 run 的最小阈值）。",
    )
    parser.add_argument("--w_request", type=float, default=0.6, help="request 权重。")
    parser.add_argument("--w_prefill", type=float, default=0.2, help="prefill 权重。")
    parser.add_argument("--w_decode", type=float, default=0.2, help="decode 权重。")
    parser.add_argument(
        "--strategy",
        choices=["max_threshold_within_regression", "min_score"],
        default="max_threshold_within_regression",
        help="选择策略：在可接受回归内取最大阈值，或直接取最小 score。",
    )
    parser.add_argument(
        "--max_request_regression",
        type=float,
        default=0.02,
        help="允许 request_ratio 的最大回归比例（例如 0.02 表示 +2%%）。",
    )
    parser.add_argument(
        "--max_prefill_regression",
        type=float,
        default=0.02,
        help="允许 prefill_ratio 的最大回归比例（例如 0.02 表示 +2%%）。",
    )
    args = parser.parse_args()

    runs = load_all_runs(args.root)
    if not runs:
        raise SystemExit(f"未找到有效数据目录: {args.root}")

    best_th, summary = select_threshold(
        runs,
        baseline_preferred=args.baseline,
        w_request=args.w_request,
        w_prefill=args.w_prefill,
        w_decode=args.w_decode,
        strategy=args.strategy,
        max_request_regression=args.max_request_regression,
        max_prefill_regression=args.max_prefill_regression,
    )

    print(f"Found runs: {len(runs)} (from {args.root})")
    headers = ["threshold", "runs", "request_ratio", "prefill_ratio", "decode_ratio", "score"]
    table_rows: List[List[str]] = []
    for th in sorted(summary.keys()):
        row = summary[th]
        table_rows.append(
            [
                str(th),
                str(int(row["runs"])),
                f"{row['request_ratio']:.4f}",
                f"{row['prefill_ratio']:.4f}",
                f"{row['decode_ratio']:.4f}",
                f"{row['score']:.4f}",
            ]
        )

    col_widths = [
        max(len(headers[i]), max((len(r[i]) for r in table_rows), default=0))
        for i in range(len(headers))
    ]
    header_line = "  ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    print(header_line)
    for r in table_rows:
        print("  ".join(r[i].rjust(col_widths[i]) for i in range(len(headers))))
    print(f"\nRecommended min_chunk_threshold = {best_th}")


if __name__ == "__main__":
    main()
