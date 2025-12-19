import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


"""
根据离线基准测试导出的 CSV，选择一个“最小分块阈值”(min_chunk_threshold)。

数据目录结构（由 offline_get_threshold.sh 生成）：
  threshold_output/<时间戳>/<threshold>/
    - request_execution_time.csv
    - request_execution_time_normalized.csv
    - prefill_e2e_time.csv
    - prefill_time_execution_plus_preemption_normalized.csv
    - decode_time_execution_plus_preemption_normalized.csv

  允许多次重复跑 bash 脚本，把每次运行当成一个 run。
  本脚本会对每个 run 先做归一化，再跨 run 聚合，从而降低偶然误差。
"""


@dataclass(frozen=True)
class ThresholdMetrics:
    """某个阈值下的汇总指标（单个 threshold、单次 run）。"""

    request_p95_s: float
    request_exec_norm_mean_s: float
    prefill_p95_s: float
    prefill_exec_norm_mean_s: float
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
    从某个 threshold 目录加载并汇总指标：
    - request_execution_time(P95)：整体 GPU 执行尾部回归约束
    - prefill_e2e_time(P95)：TTFT 尾部回归约束
    - prefill_time_execution_plus_preemption_normalized(mean)：prefill 单位 token 成本（效率代理）
    - request_execution_time_normalized(mean)：整体单位 token 纯执行成本（效率代理）
    - decode_time_execution_plus_preemption_normalized(mean)：decode 单位 token 成本（TPOT 代理）
    """

    req_csv = threshold_dir / "request_execution_time.csv"
    req_norm_csv = threshold_dir / "request_execution_time_normalized.csv"
    prefill_csv = threshold_dir / "prefill_e2e_time.csv"
    prefill_exec_norm_csv = (
        threshold_dir / "prefill_time_execution_plus_preemption_normalized.csv"
    )
    decode_csv = threshold_dir / "decode_time_execution_plus_preemption_normalized.csv"

    request_vals = _read_metric_values(req_csv, "request_execution_time")
    request_norm_vals = _read_metric_values(
        req_norm_csv, "request_execution_time_normalized"
    )
    prefill_vals = _read_metric_values(prefill_csv, "prefill_e2e_time")
    prefill_exec_norm_vals = _read_metric_values(
        prefill_exec_norm_csv, "prefill_time_execution_plus_preemption_normalized"
    )
    decode_vals = _read_metric_values(
        decode_csv, "decode_time_execution_plus_preemption_normalized"
    )

    return ThresholdMetrics(
        request_p95_s=_quantile(request_vals, 0.95),
        request_exec_norm_mean_s=mean(request_norm_vals),
        prefill_p95_s=_quantile(prefill_vals, 0.95),
        prefill_exec_norm_mean_s=mean(prefill_exec_norm_vals),
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


def _aggregate(values: List[float], agg: str) -> float:
    """跨 run 聚合：mean 默认；median 更稳健；p90/max 更保守。"""

    if not values:
        raise ValueError("values is empty")
    if agg == "mean":
        return mean(values)
    if agg == "median":
        return median(values)
    if agg == "p90":
        return _quantile(values, 0.9)
    if agg == "max":
        return max(values)
    raise ValueError(f"Unknown agg: {agg}")


def select_threshold(
    runs: Dict[str, Dict[int, ThresholdMetrics]],
    baseline_preferred: int = 0,
    w_request: float = 0.35,
    w_prefill: float = 0.35,
    w_prefill_exec_norm: float = 0.2,
    w_request_exec_norm: float = 0.05,
    w_decode: float = 0.05,
    run_agg: str = "mean",
) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    根据多次 run 的结果选择阈值。

    关键做法：每个 run 内先按 baseline 做归一化（ratio），再跨 run 聚合（mean/median/p90/max）。
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
            per_threshold.setdefault(th, {}).setdefault(
                "request_exec_norm_ratio", []
            ).append(m.request_exec_norm_mean_s / base.request_exec_norm_mean_s)
            per_threshold.setdefault(th, {}).setdefault("prefill_ratio", []).append(
                m.prefill_p95_s / base.prefill_p95_s
            )
            per_threshold.setdefault(th, {}).setdefault(
                "prefill_exec_norm_ratio", []
            ).append(m.prefill_exec_norm_mean_s / base.prefill_exec_norm_mean_s)
            per_threshold.setdefault(th, {}).setdefault("decode_ratio", []).append(
                m.decode_mean_s / base.decode_mean_s
            )

    summary: Dict[int, Dict[str, float]] = {}
    for th, ratios in per_threshold.items():
        if not (
            ratios.get("request_ratio")
            and ratios.get("request_exec_norm_ratio")
            and ratios.get("prefill_ratio")
            and ratios.get("prefill_exec_norm_ratio")
            and ratios.get("decode_ratio")
        ):
            continue
        req = _aggregate(ratios["request_ratio"], run_agg)
        req_norm = _aggregate(ratios["request_exec_norm_ratio"], run_agg)
        pre = _aggregate(ratios["prefill_ratio"], run_agg)
        pre_norm = _aggregate(ratios["prefill_exec_norm_ratio"], run_agg)
        dec = _aggregate(ratios["decode_ratio"], run_agg)
        score = (
            w_request * req
            + w_prefill * pre
            + w_prefill_exec_norm * pre_norm
            + w_request_exec_norm * req_norm
            + w_decode * dec
        )
        summary[th] = {
            "runs": float(len(ratios["request_ratio"])),
            "request_ratio": req,
            "request_exec_norm_ratio": req_norm,
            "prefill_ratio": pre,
            "prefill_exec_norm_ratio": pre_norm,
            "decode_ratio": dec,
            # score 越小越好；权重默认更偏向 request 尾延迟。
            "score": score,
        }

    if not summary:
        raise RuntimeError("No valid threshold data found under threshold_output/")

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
    parser.add_argument("--w_request", type=float, default=0.35, help="request 权重。")
    parser.add_argument("--w_prefill", type=float, default=0.35, help="prefill 权重。")
    parser.add_argument(
        "--w_prefill_exec_norm",
        type=float,
        default=0.2,
        help="prefill_exec_norm_ratio 权重（prefill 单位 token 成本的代理）。",
    )
    parser.add_argument(
        "--w_request_exec_norm",
        type=float,
        default=0.05,
        help="request_exec_norm_ratio 权重（整体单位 token 纯执行成本的代理）。",
    )
    parser.add_argument("--w_decode", type=float, default=0.05, help="decode 权重。")
    parser.add_argument(
        "--run_agg",
        choices=["mean", "median", "p90", "max"],
        default="mean",
        help="跨多个 run 聚合 ratio 的方式：mean/median/p90/max（越保守越偏向 p90/max）。",
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
        w_prefill_exec_norm=args.w_prefill_exec_norm,
        w_request_exec_norm=args.w_request_exec_norm,
        w_decode=args.w_decode,
        run_agg=args.run_agg,
    )

    print(f"Found runs: {len(runs)} (from {args.root})")
    headers = [
        "threshold",
        "runs",
        "request_ratio",
        "request_exec_norm_ratio",
        "prefill_ratio",
        "prefill_exec_norm_ratio",
        "decode_ratio",
        "score",
    ]
    table_rows: List[List[str]] = []
    for th in sorted(summary.keys()):
        row = summary[th]
        table_rows.append(
            [
                str(th),
                str(int(row["runs"])),
                f"{row['request_ratio']:.4f}",
                f"{row['request_exec_norm_ratio']:.4f}",
                f"{row['prefill_ratio']:.4f}",
                f"{row['prefill_exec_norm_ratio']:.4f}",
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
