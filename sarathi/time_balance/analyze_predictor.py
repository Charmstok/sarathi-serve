from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import List

import torch

from sarathi.time_balance.predict_time import CSV_PATH, MODEL_CACHE_PATH, TimePredictor


@dataclass(frozen=True)
class SelectStats:
    decode_tokens: torch.Tensor
    decode_history_tokens: torch.Tensor
    batch_request_count: torch.Tensor
    prefill_tokens: torch.Tensor
    prefill_processed_tokens: torch.Tensor
    latency_ms: torch.Tensor


def _read_select_stats_csv(path: str) -> SelectStats:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}. Update `CSV_PATH` in sarathi/time_balance/predict_time.py."
        )

    decode_tokens: List[float] = []
    decode_history_tokens: List[float] = []
    batch_request_count: List[float] = []
    prefill_tokens: List[float] = []
    prefill_processed_tokens: List[float] = []
    latency_ms: List[float] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "decode_tokens",
            "decode_history_tokens",
            "batch_request_count",
            "prefill_tokens",
            "prefill_processed_tokens",
            "latency_ms",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV missing columns {sorted(missing)}: {path}. Got columns: {reader.fieldnames}"
            )

        for row in reader:
            decode_tokens.append(float(row["decode_tokens"]))
            decode_history_tokens.append(float(row["decode_history_tokens"]))
            batch_request_count.append(float(row["batch_request_count"]))
            prefill_tokens.append(float(row["prefill_tokens"]))
            prefill_processed_tokens.append(float(row["prefill_processed_tokens"]))
            latency_ms.append(float(row["latency_ms"]))

    return SelectStats(
        decode_tokens=torch.tensor(decode_tokens, dtype=torch.float32),
        decode_history_tokens=torch.tensor(decode_history_tokens, dtype=torch.float32),
        batch_request_count=torch.tensor(batch_request_count, dtype=torch.float32),
        prefill_tokens=torch.tensor(prefill_tokens, dtype=torch.float32),
        prefill_processed_tokens=torch.tensor(prefill_processed_tokens, dtype=torch.float32),
        latency_ms=torch.tensor(latency_ms, dtype=torch.float32),
    )


def _percentile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, torch.tensor(q, dtype=x.dtype)).item())


def _summary_line(prefix: str, y_true: torch.Tensor) -> str:
    if y_true.numel() == 0:
        return f"{prefix}: n=0"
    mean = float(y_true.mean().item())
    std = float(y_true.std(unbiased=False).item())
    return (
        f"{prefix}: n={y_true.numel()} mean={mean:.3f} std={std:.3f} "
        f"min={float(y_true.min().item()):.3f} max={float(y_true.max().item()):.3f}"
    )


def _error_report(name: str, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
    if y_true.numel() == 0:
        print(f"[{name}] n=0")
        return

    err = y_pred - y_true
    abs_err = err.abs()
    mae = float(abs_err.mean().item())
    rmse = math.sqrt(float((err * err).mean().item()))

    p50 = _percentile(abs_err, 0.50)
    p90 = _percentile(abs_err, 0.90)
    p95 = _percentile(abs_err, 0.95)
    p99 = _percentile(abs_err, 0.99)

    print(
        f"[{name}] n={y_true.numel()} mae={mae:.3f} rmse={rmse:.3f} "
        f"p50={p50:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f}"
    )


def _topk_worst(
    stats: SelectStats,
    y_pred: torch.Tensor,
    k: int = 10,
    *,
    mask: torch.Tensor | None = None,
    title: str = "worst",
) -> None:
    y_true = stats.latency_ms
    if mask is None:
        mask = torch.ones_like(y_true, dtype=torch.bool)

    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        print(f"[{title}] n=0")
        return

    abs_err = (y_pred[idx] - y_true[idx]).abs()
    topk = min(int(k), int(abs_err.numel()))
    values, order = torch.topk(abs_err, k=topk, largest=True)

    print(f"[{title}] top{topk} by abs error")
    for rank in range(topk):
        i = int(idx[int(order[rank])].item())
        print(
            f"  #{rank+1}: abs_err={float(values[rank].item()):.3f} "
            f"pred={float(y_pred[i].item()):.3f} true={float(y_true[i].item()):.3f} "
            f"decode={int(stats.decode_tokens[i].item())} "
            f"dec_hist={int(stats.decode_history_tokens[i].item())} "
            f"batch_req={int(stats.batch_request_count[i].item())} "
            f"prefill={int(stats.prefill_tokens[i].item())} "
            f"prefill_processed={int(stats.prefill_processed_tokens[i].item())}"
        )


def main() -> None:
    csv_path = CSV_PATH
    stats = _read_select_stats_csv(csv_path)
    print(f"csv={csv_path}")
    print(_summary_line("latency_ms", stats.latency_ms))

    if os.path.exists(MODEL_CACHE_PATH):
        predictor = TimePredictor.load(MODEL_CACHE_PATH)
        print(f"model=loaded {MODEL_CACHE_PATH}")
    else:
        predictor = TimePredictor.from_csv(csv_path, verbose=True)
        print(f"model=trained_from_csv (cache missing: {MODEL_CACHE_PATH})")

    y_pred = predictor.predict(
        decode_tokens=stats.decode_tokens,
        decode_history_tokens=stats.decode_history_tokens,
        batch_request_count=stats.batch_request_count,
        prefill_tokens=stats.prefill_tokens,
        prefill_processed_tokens=stats.prefill_processed_tokens,
    ).to(dtype=torch.float32)

    mask_prefill = stats.prefill_tokens > 0
    mask_decode_only = stats.prefill_tokens == 0

    _error_report("overall", stats.latency_ms, y_pred)
    _error_report(
        "decode_only(prefill=0)",
        stats.latency_ms[mask_decode_only],
        y_pred[mask_decode_only],
    )
    _error_report(
        "prefill(prefill>0)", stats.latency_ms[mask_prefill], y_pred[mask_prefill]
    )

    _topk_worst(stats, y_pred, k=10, title="overall_worst")
    _topk_worst(stats, y_pred, k=10, mask=mask_prefill, title="prefill_worst")


if __name__ == "__main__":
    main()

