from __future__ import annotations

import argparse
import json
from pathlib import Path


import torch

from sarathi.time_balance.config import MODEL_CACHE_PATH, TEST_CSV_PATH
from sarathi.time_balance.predict_time import TimePredictor, _read_select_stats_csv


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

CHINESE_FONT_SIZE = 10.5
ENGLISH_FONT_SIZE = 10.5
TITLE_FONT_SIZE = 10.5
TICK_FONT_SIZE = 10.5
LEGEND_FONT_SIZE = 10.5
ANNOTATION_FONT_SIZE = 9.5

SIMSUN_FONT_PATH = Path("/usr/share/fonts/truetype/simsun/SIMSUN.ttf")
TIMES_NEW_ROMAN_FONT_PATH = Path(
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
)

SERIF_FONT_CANDIDATES = [
    "Times New Roman",
    "SimSun",
    "DejaVu Serif",
]


def register_paper_fonts() -> None:
    try:
        from matplotlib import font_manager
    except ImportError:
        return

    for font_path in [SIMSUN_FONT_PATH, TIMES_NEW_ROMAN_FONT_PATH]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def get_plot_modules():
    try:
        import matplotlib

        matplotlib.use("Agg")
        register_paper_fonts()
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc
    return plt


def configure_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": SERIF_FONT_CANDIDATES,
            "font.serif": SERIF_FONT_CANDIDATES,
            "axes.linewidth": 0.9,
            "axes.labelsize": CHINESE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def get_cjk_serif_font_properties():
    try:
        from matplotlib import font_manager
    except ImportError:
        return None

    if SIMSUN_FONT_PATH.exists():
        return font_manager.FontProperties(fname=str(SIMSUN_FONT_PATH))

    for font_name in CJK_SERIF_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
        except Exception:
            continue
        if font_path:
            return font_manager.FontProperties(fname=font_path)
    return None


def get_latin_serif_font_properties():
    try:
        from matplotlib import font_manager
    except ImportError:
        return None

    if TIMES_NEW_ROMAN_FONT_PATH.exists():
        return font_manager.FontProperties(fname=str(TIMES_NEW_ROMAN_FONT_PATH))

    for font_name in LATIN_SERIF_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(font_name, fallback_to_default=False)
        except Exception:
            continue
        if font_path:
            return font_manager.FontProperties(fname=font_path)
    return None


def get_axis_label_font_properties():
    font = get_cjk_serif_font_properties()
    if font is not None:
        font.set_size(CHINESE_FONT_SIZE)
    return font


def get_title_font_properties():
    font = get_cjk_serif_font_properties()
    if font is not None:
        font.set_size(TITLE_FONT_SIZE)
    return font


def get_legend_font_properties():
    font = get_cjk_serif_font_properties()
    if font is not None:
        font.set_size(LEGEND_FONT_SIZE)
    return font


def get_annotation_font_properties():
    font = get_latin_serif_font_properties()
    if font is not None:
        font.set_size(ANNOTATION_FONT_SIZE)
    return font


def set_plot_title(ax, title: str) -> None:
    title_font = get_title_font_properties()
    if title_font is not None:
        ax.set_title(title, pad=6, fontproperties=title_font)
    else:
        ax.set_title(title, pad=6)


def set_legend(ax, loc: str = "best") -> None:
    ax.legend(loc=loc, frameon=False, handlelength=1.8)


def apply_paper_axes(ax) -> None:
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.28)
    ax.tick_params(axis="both", which="major", pad=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MLP time predictor and generate paper-ready plots."
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_CACHE_PATH,
        help="Path to the saved predictor checkpoint.",
    )
    parser.add_argument(
        "--csv-path",
        default=TEST_CSV_PATH,
        help="CSV used for evaluation. Defaults to the configured test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_plots" / "time_predictor_eval",
        help="Directory used to store metrics and plots.",
    )
    parser.add_argument(
        "--figure-prefix",
        default="time_predictor_test",
        help="Stem used for output files.",
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=5000,
        help="Maximum number of points rendered in the scatter plot.",
    )
    return parser.parse_args()


def evenly_sample_points(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actual.numel() <= max_points:
        return actual, predicted

    indices = torch.linspace(
        0,
        actual.numel() - 1,
        steps=max_points,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    return actual[indices], predicted[indices]


def compute_metrics(actual: torch.Tensor, predicted: torch.Tensor) -> dict[str, float]:
    error = predicted - actual
    abs_error = error.abs()
    squared_error = error.square()
    safe_actual = actual.abs().clamp_min(1e-6)

    metrics = {
        "count": float(actual.numel()),
        "mae_ms": float(abs_error.mean().item()),
        "rmse_ms": float(torch.sqrt(squared_error.mean()).item()),
        "mape_pct": float((abs_error / safe_actual).mean().mul(100.0).item()),
        "p50_abs_error_ms": float(torch.quantile(abs_error, 0.50).item()),
        "p90_abs_error_ms": float(torch.quantile(abs_error, 0.90).item()),
        "p95_abs_error_ms": float(torch.quantile(abs_error, 0.95).item()),
        "p99_abs_error_ms": float(torch.quantile(abs_error, 0.99).item()),
        "acc_at_5ms": float((abs_error <= 5.0).to(torch.float32).mean().item()),
        "acc_at_10ms": float((abs_error <= 10.0).to(torch.float32).mean().item()),
        "acc_at_10pct": float(
            ((abs_error / safe_actual) <= 0.10).to(torch.float32).mean().item()
        ),
        "mean_signed_error_ms": float(error.mean().item()),
        "underpredict_rate": float((error < 0).to(torch.float32).mean().item()),
    }

    under_errors = (-error[error < 0]).to(torch.float32)
    over_errors = error[error > 0].to(torch.float32)
    metrics["mean_underpredict_ms"] = (
        float(under_errors.mean().item()) if under_errors.numel() > 0 else 0.0
    )
    metrics["mean_overpredict_ms"] = (
        float(over_errors.mean().item()) if over_errors.numel() > 0 else 0.0
    )
    return metrics


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def plot_predicted_vs_actual(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    output_path: Path,
    max_points: int,
) -> None:
    plt = get_plot_modules()
    configure_plot_style(plt)

    sampled_actual, sampled_predicted = evenly_sample_points(
        actual,
        predicted,
        max_points=max_points,
    )
    lo = min(float(sampled_actual.min().item()), float(sampled_predicted.min().item()))
    hi = max(float(sampled_actual.max().item()), float(sampled_predicted.max().item()))
    pad = max(2.0, (hi - lo) * 0.05)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(
        sampled_actual.numpy(),
        sampled_predicted.numpy(),
        s=12,
        alpha=0.38,
        color="#2f5d8c",
        edgecolors="none",
        rasterized=True,
        label="样本点",
    )
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        linewidth=1.0,
        color="#b33a3a",
        label="理想参考线 y=x",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("真实延迟（毫秒）")
    ax.set_ylabel("预测延迟（毫秒）")
    set_plot_title(ax, "预测值与真实值对比")
    apply_paper_axes(ax)
    ax.set_aspect("equal", adjustable="box")
    set_legend(ax, loc="upper left")
    fig.tight_layout(pad=1.1)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def plot_absolute_error_cdf(
    actual: torch.Tensor,
    predicted: torch.Tensor,
    output_path: Path,
) -> None:
    plt = get_plot_modules()
    configure_plot_style(plt)

    abs_error = (predicted - actual).abs()
    sorted_error, _ = torch.sort(abs_error)
    cdf = torch.arange(1, sorted_error.numel() + 1, dtype=torch.float32)
    cdf = cdf / float(sorted_error.numel())

    max_error = float(sorted_error[-1].item())
    x_max = max_error * 1.05 if max_error > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.step(
        sorted_error.numpy(),
        cdf.numpy(),
        where="post",
        color="#2b7b43",
        linewidth=1.6,
        label="绝对误差 CDF",
    )
    for threshold, color in [(5.0, "#d62728")]:
        accuracy = float((abs_error <= threshold).to(torch.float32).mean().item())
        if threshold <= x_max:
            ax.axvline(
                threshold,
                linestyle=(0, (5, 3)),
                linewidth=0.9,
                color=color,
                alpha=0.8,
                label=f"<= {threshold:.0f} ms: {accuracy:.1%}",
            )
        ax.axhline(
            accuracy,
            linestyle=(0, (5, 3)),
            linewidth=0.9,
            color=color,
            alpha=0.8,
        )

    quantile_styles = [
        (0.50, "P50", "o"),
        (0.90, "P90", "s"),
        (0.95, "P95", "^"),
        (0.99, "P99", "D"),
    ]
    quantile_color = "#000000"
    quantile_handles = []
    quantile_labels = []
    for quantile, label, marker in quantile_styles:
        value = float(torch.quantile(abs_error, quantile).item())
        ax.axvline(
            value,
            ymin=0.0,
            ymax=quantile,
            linestyle=":",
            linewidth=0.7,
            color=quantile_color,
            alpha=0.45,
        )
        handle = ax.scatter(
            [value],
            [quantile],
            s=30,
            marker=marker,
            facecolor=quantile_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label="_nolegend_",
        )
        quantile_handles.append(handle)
        quantile_labels.append(f"{label}={value:.2f} ms")

    quantile_legend = ax.legend(
        handles=quantile_handles,
        labels=quantile_labels,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="#d9d9d9",
        facecolor="white",
        fontsize=ANNOTATION_FONT_SIZE,
        borderpad=0.35,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    ax.add_artist(quantile_legend)
    ax.set_xlabel("绝对误差（毫秒）")
    ax.set_ylabel("累积分布函数")
    set_plot_title(ax, "绝对误差累积分布")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, 1.005)
    apply_paper_axes(ax)
    set_legend(ax, loc="lower right")
    fig.tight_layout(pad=1.1)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".png"))
    plt.close(fig)


def print_metrics(metrics: dict[str, float]) -> None:
    print(f"count               : {int(metrics['count'])}")
    print(f"mae_ms              : {metrics['mae_ms']:.4f}")
    print(f"rmse_ms             : {metrics['rmse_ms']:.4f}")
    print(f"mape_pct            : {metrics['mape_pct']:.4f}")
    print(f"p50_abs_error_ms    : {metrics['p50_abs_error_ms']:.4f}")
    print(f"p90_abs_error_ms    : {metrics['p90_abs_error_ms']:.4f}")
    print(f"p95_abs_error_ms    : {metrics['p95_abs_error_ms']:.4f}")
    print(f"p99_abs_error_ms    : {metrics['p99_abs_error_ms']:.4f}")
    print(f"acc_at_5ms          : {metrics['acc_at_5ms']:.4%}")
    print(f"acc_at_10ms         : {metrics['acc_at_10ms']:.4%}")
    print(f"acc_at_10pct        : {metrics['acc_at_10pct']:.4%}")
    print(f"mean_signed_error_ms: {metrics['mean_signed_error_ms']:.4f}")
    print(f"underpredict_rate   : {metrics['underpredict_rate']:.4%}")
    print(f"mean_underpredict_ms: {metrics['mean_underpredict_ms']:.4f}")
    print(f"mean_overpredict_ms : {metrics['mean_overpredict_ms']:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = TimePredictor.load(args.model_path)
    features, actual = _read_select_stats_csv(args.csv_path)
    predicted = predictor.predict_from_features(features.to(dtype=torch.float32))
    actual = actual.to(dtype=torch.float32)

    metrics = compute_metrics(actual, predicted)
    metrics_path = output_dir / f"{args.figure_prefix}_metrics.json"
    save_metrics(metrics, metrics_path)

    scatter_path = output_dir / f"{args.figure_prefix}_pred_vs_actual"
    error_cdf_path = output_dir / f"{args.figure_prefix}_abs_error_cdf"
    plot_predicted_vs_actual(
        actual,
        predicted,
        scatter_path,
        max_points=max(100, args.scatter_max_points),
    )
    plot_absolute_error_cdf(actual, predicted, error_cdf_path)

    print(f"model_path          : {args.model_path}")
    print(f"csv_path            : {args.csv_path}")
    print(f"metrics_path        : {metrics_path}")
    print(
        "scatter_figure      : "
        f"{scatter_path.with_suffix('.pdf')} / {scatter_path.with_suffix('.png')}"
    )
    print(
        "error_cdf_figure    : "
        f"{error_cdf_path.with_suffix('.pdf')} / {error_cdf_path.with_suffix('.png')}"
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
