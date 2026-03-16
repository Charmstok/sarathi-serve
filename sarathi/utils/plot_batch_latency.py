import csv
import math
from pathlib import Path


CSV_RELATIVE_PATH = (
    "offline_inference_output/"
    "2026-03-13_15-12-25-时间预算-100ms/"
    #"2026-03-13_15-21-48-token预算-256tokens/"
    "replica_0/select_stats_rank0.csv"
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

    batch_indices = list(range(len(latency_values)))
    output_dir = csv_path.parent / "plot_batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_latency_ms.png"
    y_min, y_max = get_latency_axis_limits(latency_values)

    plt.figure(figsize=(12, 6))
    plt.plot(batch_indices, latency_values, linewidth=1.5)
    plt.xlabel("Batch Index")
    plt.ylabel("Latency (ms)")
    plt.title("Batch Latency")
    plt.ylim(y_min, y_max)
    plt.yticks(range(int(y_min), int(y_max) + 1, 10))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def bucketize_latency(latency_values: list[float]) -> tuple[list[str], list[int]]:
    labels = [
        "0-10ms",
        "10-20ms",
        "20-30ms",
        "30-40ms",
        "40-50ms",
        "50-60ms",
        "60-70ms",
        "70-80ms",
        "80-90ms",
        "90-100ms",
        ">100ms",
    ]
    counts = [0] * len(labels)

    for latency in latency_values:
        if latency < 10:
            counts[0] += 1
        elif latency < 20:
            counts[1] += 1
        elif latency < 30:
            counts[2] += 1
        elif latency < 40:
            counts[3] += 1
        elif latency > 100:
            counts[-1] += 1
        elif latency < 50:
            counts[4] += 1
        elif latency < 60:
            counts[5] += 1
        elif latency < 70:
            counts[6] += 1
        elif latency < 80:
            counts[7] += 1
        elif latency < 90:
            counts[8] += 1
        else:
            counts[9] += 1

    filtered_labels = []
    filtered_counts = []
    for label, count in zip(labels, counts):
        if count > 0:
            filtered_labels.append(label)
            filtered_counts.append(count)

    return filtered_labels, filtered_counts


def plot_latency_bar(csv_path: Path, latency_values: list[float]) -> Path:
    plt = get_plot_modules()
    labels, counts = bucketize_latency(latency_values)
    output_dir = csv_path.parent / "plot_batch"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{csv_path.stem}_latency_ms_bar.png"

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, counts, width=0.7, color="#4C72B0")
    plt.xlabel("Latency Range")
    plt.ylabel("Batch Count")
    plt.title("Batch Latency Distribution")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    max_count = max(counts)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(0.2, max_count * 0.01),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

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
    bar_chart_path = plot_latency_bar(csv_path, latency_values)
    print(f"CSV 路径: {csv_path}")
    print(f"折线图已保存: {line_chart_path}")
    print(f"柱状图已保存: {bar_chart_path}")


if __name__ == "__main__":
    main()
