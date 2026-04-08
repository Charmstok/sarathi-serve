from __future__ import annotations

import argparse
from pathlib import Path


FIGURE_NAME = "prefill_e2e_time(时间预算)"
TITLE = "prefill_e2e_time（时间预算）"
X_LABEL = "Average Inter-Arrival Time (s)"
Y_LABEL = "Latency (ms)"

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

# Each value is the average interval between two arriving requests.
INTER_ARRIVAL_SECONDS = [0.1, 0.15, 0.3, 0.5, 1.0, 1.5]

# Replace the values below with your measured percentiles.
# Each list must align one-to-one with INTER_ARRIVAL_SECONDS.
P99_MS = [80, 82, 84, 85, 86, 88]
P90_MS = [
    # e.g. 72, 74, 76, 78, 79, 80
]
P80_MS = [
    # e.g. 68, 70, 72, 74, 75, 76
]
P50_MS = [
    # e.g. 55, 57, 59, 60, 61, 62
]


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


def get_mixed_serif_font_properties():
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    return font_manager.FontProperties(family=SERIF_FONT_CANDIDATES)


def build_latency_series() -> dict[str, tuple[list[float], list[float]]]:
    if not INTER_ARRIVAL_SECONDS:
        raise ValueError("INTER_ARRIVAL_SECONDS must not be empty.")

    base_x = list(INTER_ARRIVAL_SECONDS)
    candidates = {
        "p99": P99_MS,
        "p90": P90_MS,
        "p80": P80_MS,
        "p50": P50_MS,
    }

    series: dict[str, tuple[list[float], list[float]]] = {}
    for label, values in candidates.items():
        if not values:
            continue
        if len(values) != len(base_x):
            raise ValueError(
                f"{label} has {len(values)} values, but "
                f"INTER_ARRIVAL_SECONDS has {len(base_x)} values."
            )
        ordered_pairs = sorted(zip(base_x, values), key=lambda item: item[0])
        x_values = [item[0] for item in ordered_pairs]
        y_values = [item[1] for item in ordered_pairs]
        series[label] = (x_values, y_values)

    if not series:
        raise ValueError("At least one percentile series must be filled before plotting.")

    return series


def get_y_limits(series: dict[str, tuple[list[float], list[float]]]) -> tuple[float, float]:
    y_values = [value for _, values in series.values() for value in values]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        padding = max(1.0, y_min * 0.05)
        return y_min - padding, y_max + padding

    padding = max(2.0, (y_max - y_min) * 0.12)
    return y_min - padding, y_max + padding


def plot_percentiles(output_dir: Path, figure_name: str, title: str | None) -> tuple[Path, Path]:
    plt = get_plot_modules()
    configure_plot_style(plt)
    series = build_latency_series()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{figure_name}.pdf"
    png_path = output_dir / f"{figure_name}.png"

    colors = {
        "p99": "#1f77b4",
        "p90": "#ff7f0e",
        "p80": "#2ca02c",
        "p50": "#d62728",
    }
    markers = {
        "p99": "o",
        "p90": "s",
        "p80": "^",
        "p50": "D",
    }

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label in ["p99", "p90", "p80", "p50"]:
        if label not in series:
            continue
        x_values, y_values = series[label]
        ax.plot(
            x_values,
            y_values,
            label=label.upper(),
            color=colors[label],
            marker=markers[label],
            linewidth=2.0,
            markersize=6.0,
        )

    x_ticks = sorted(INTER_ARRIVAL_SECONDS)
    y_min, y_max = get_y_limits(series)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:g}" for tick in x_ticks], rotation=25, ha="right")
    ax.tick_params(axis="x", pad=6)
    ax.set_ylim(y_min, y_max)
    ax.margins(x=0.04)

    if title:
        title_font = get_mixed_serif_font_properties()
        if title_font is not None:
            ax.set_title(title, pad=10, fontproperties=title_font)
        else:
            ax.set_title(title, pad=10)

    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#d9d9d9")

    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot percentile latency curves against inter-arrival time."
    )
    parser.add_argument(
        "--figure-name",
        default=FIGURE_NAME,
        help="Output file stem, e.g. prefill_e2e_time.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "paper_plots",
        help="Directory used to store the generated figure files.",
    )
    parser.add_argument(
        "--title",
        default=TITLE,
        help="Optional figure title. Leave unset for paper-style plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path, png_path = plot_percentiles(args.output_dir, args.figure_name, args.title)
    print(f"Figure saved to: {pdf_path}")
    print(f"Figure saved to: {png_path}")


if __name__ == "__main__":
    main()
