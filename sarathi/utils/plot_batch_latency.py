import json
import csv
import math
from pathlib import Path
from typing import Optional


TARGET_LATENCY_MS: Optional[float] = None

CSV_RELATIVE_PATH = (
    "offline_inference_output/"
    #"2026-04-13_17-54-58-时间预算-105ms/"

    "2026-04-01_14-31-21-token预算-1024tokens（对比2）/"
    "replica_0/select_stats_rank0.csv"
)

LATIN_SERIF_FONT_CANDIDATES = [
    "Times New Roman",
    "Liberation Serif",
]

CJK_SERIF_FONT_CANDIDATES = [
    "SimSun",
    "Songti SC",
    "Noto Serif CJK SC",
    "AR PL UMing CN",
    "Noto Serif CJK JP",
]

SERIF_FONT_CANDIDATES = (
    LATIN_SERIF_FONT_CANDIDATES
    + CJK_SERIF_FONT_CANDIDATES
    + ["DejaVu Serif"]
)


def get_plot_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "缺少 matplotlib，请先安装：`pip install matplotlib`"
        ) from exc
    return plt


def configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_FONT_CANDIDATES,
            "axes.linewidth": 1.0,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_cjk_serif_font_properties():
    try:
        from matplotlib import font_manager
    except ImportError:
        return None

    for font_name in CJK_SERIF_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
        except Exception:
            continue
        if font_path:
            return font_manager.FontProperties(fname=font_path)
    return None


def set_axis_labels(ax, xlabel: str, ylabel: str) -> None:
    cjk_font = get_cjk_serif_font_properties()
    if cjk_font is not None:
        ax.set_xlabel(xlabel, fontproperties=cjk_font)
        ax.set_ylabel(ylabel, fontproperties=cjk_font)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


def set_title(ax, title: str) -> None:
    cjk_font = get_cjk_serif_font_properties()
    if cjk_font is not None:
        ax.set_title(title, pad=10, fontproperties=cjk_font)
    else:
        ax.set_title(title, pad=10)


def set_legend(ax) -> None:
    cjk_font = get_cjk_serif_font_properties()
    legend_kwargs = {
        "loc": "lower left",
        "bbox_to_anchor": (0.01, 0.01),
        "frameon": True,
        "framealpha": 0.95,
        "edgecolor": "#d9d9d9",
    }
    if cjk_font is not None:
        legend_kwargs["prop"] = cjk_font
        ax.legend(**legend_kwargs)
    else:
        ax.legend(**legend_kwargs)


def load_latency_values(csv_path: Path) -> list[float]:
    latency_values: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "latency_ms" not in (reader.fieldnames or []):
            raise ValueError("CSV 文件中不存在 `latency_ms` 字段。")

        for row in reader:
            latency = row.get("latency_ms", "").strip()
            if not latency:
                continue
            latency_values.append(float(latency))
    return latency_values


def get_latency_axis_limits(latency_values: list[float]) -> tuple[float, float]:
    max_latency = max(latency_values)
    y_min = 0.0
    y_max = max(110.0, math.ceil(max_latency / 10.0) * 10.0)
    return y_min, y_max


def plot_latency(csv_path: Path, latency_values: list[float]) -> Path:
    plt = get_plot_modules()
    configure_plot_style(plt)

    batch_indices = list(range(len(latency_values)))
    output_dir = csv_path.parent / "plot_batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_latency_ms.png"
    y_min, y_max = get_latency_axis_limits(latency_values)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(
        batch_indices,
        latency_values,
        color="#4C72B0",
        s=16,
        alpha=0.75,
        edgecolors="none",
        label="批次执行时间",
    )
    if TARGET_LATENCY_MS is not None:
        ax.axhline(
            TARGET_LATENCY_MS,
            color="#D62728",
            linestyle="--",
            linewidth=1.6,
            label=f"目标时间 {TARGET_LATENCY_MS:g}ms",
        )
        ax.axhspan(
            TARGET_LATENCY_MS - 5.0,
            TARGET_LATENCY_MS + 5.0,
            color="#F2C14E",
            alpha=0.18,
            label="目标 ±5ms 区间",
        )
    set_axis_labels(ax, "批次序号", "执行时间（毫秒）")
    set_title(ax, "批次执行时间散点图")
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(range(int(y_min), int(y_max) + 1, 10))
    ax.grid(True, linestyle="--", alpha=0.4)
    set_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty.")
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]

    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def build_latency_summary(
    latency_values: list[float],
    *,
    target_latency_ms: Optional[float],
) -> dict[str, float | int | None]:
    sorted_values = sorted(latency_values)
    count = len(sorted_values)
    mean_latency = sum(sorted_values) / count
    variance = sum((value - mean_latency) ** 2 for value in sorted_values) / count

    summary: dict[str, float | int | None] = {
        "样本数量": count,
        "目标执行时间_毫秒": target_latency_ms,
        "最小执行时间_毫秒": sorted_values[0],
        "最大执行时间_毫秒": sorted_values[-1],
        "平均执行时间_毫秒": mean_latency,
        "执行时间标准差_毫秒": math.sqrt(variance),
        "P50执行时间_毫秒": compute_percentile(sorted_values, 50.0),
        "P90执行时间_毫秒": compute_percentile(sorted_values, 90.0),
        "P95执行时间_毫秒": compute_percentile(sorted_values, 95.0),
        "P99执行时间_毫秒": compute_percentile(sorted_values, 99.0),
    }

    if target_latency_ms is not None:
        within_2 = sum(abs(value - target_latency_ms) <= 2.0 for value in sorted_values)
        within_5 = sum(abs(value - target_latency_ms) <= 5.0 for value in sorted_values)
        within_10 = sum(abs(value - target_latency_ms) <= 10.0 for value in sorted_values)
        below_target = sum(value < target_latency_ms for value in sorted_values)
        above_target = sum(value > target_latency_ms for value in sorted_values)
        summary.update(
            {
                "落在目标时间正负2毫秒内的批次数量": within_2,
                "落在目标时间正负2毫秒内的批次占比": within_2 / count,
                "落在目标时间正负5毫秒内的批次数量": within_5,
                "落在目标时间正负5毫秒内的批次占比": within_5 / count,
                "落在目标时间正负10毫秒内的批次数量": within_10,
                "落在目标时间正负10毫秒内的批次占比": within_10 / count,
                "低于目标时间的批次数量": below_target,
                "低于目标时间的批次占比": below_target / count,
                "高于目标时间的批次数量": above_target,
                "高于目标时间的批次占比": above_target / count,
            }
        )

    return summary


def save_latency_summary(csv_path: Path, latency_values: list[float]) -> Path:
    output_dir = csv_path.parent / "plot_batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_latency_summary.json"
    summary = build_latency_summary(
        latency_values,
        target_latency_ms=TARGET_LATENCY_MS,
    )
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / CSV_RELATIVE_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_path}")

    latency_values = load_latency_values(csv_path)
    if not latency_values:
        raise ValueError("CSV 中没有可用的 `latency_ms` 数据。")

    line_chart_path = plot_latency(csv_path, latency_values)
    summary_json_path = save_latency_summary(csv_path, latency_values)
    print(f"CSV 路径: {csv_path}")
    print(f"散点图已保存: {line_chart_path}")
    print(f"统计结果已保存: {summary_json_path}")


if __name__ == "__main__":
    main()
