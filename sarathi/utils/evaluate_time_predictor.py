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
            "matplotlib is required. Install it with: pip install matplotlib"
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


def set_plot_title(ax, title: str) -> None:
    title_font = get_cjk_serif_font_properties()
    if title_font is not None:
        ax.set_title(title, pad=10, fontproperties=title_font)
    else:
        ax.set_title(title, pad=10)


def set_legend(ax) -> None:
    legend_font = get_cjk_serif_font_properties()
    if legend_font is not None:
        ax.legend(
            frameon=True,
            framealpha=0.95,
            edgecolor="#d9d9d9",
            prop=legend_font,
        )
    else:
        ax.legend(frameon=True, framealpha=0.95, edgecolor="#d9d9d9")


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

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(
        sampled_actual.numpy(),
        sampled_predicted.numpy(),
        s=16,
        alpha=0.45,
        color="#1f77b4",
        edgecolors="none",
        label="样本点",
    )
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        linewidth=1.3,
        color="#d62728",
        label="理想参考线",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    axis_label_font = get_cjk_serif_font_properties()
    if axis_label_font is not None:
        ax.set_xlabel("真实延迟（毫秒）", fontproperties=axis_label_font)
        ax.set_ylabel("预测延迟（毫秒）", fontproperties=axis_label_font)
    else:
        ax.set_xlabel("真实延迟（毫秒）")
        ax.set_ylabel("预测延迟（毫秒）")
    set_plot_title(ax, "预测值-真实值散点图")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    set_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(
        sorted_error.numpy(),
        cdf.numpy(),
        color="#2ca02c",
        linewidth=2.0,
    )
    for threshold, color in [(5.0, "#7f7f7f"), (10.0, "#ff7f0e")]:
        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.8,
        )
    axis_label_font = get_cjk_serif_font_properties()
    if axis_label_font is not None:
        ax.set_xlabel("绝对误差（毫秒）", fontproperties=axis_label_font)
        ax.set_ylabel("累积分布函数", fontproperties=axis_label_font)
    else:
        ax.set_xlabel("绝对误差（毫秒）")
        ax.set_ylabel("累积分布函数")
    set_plot_title(ax, "绝对误差累积分布图")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
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
