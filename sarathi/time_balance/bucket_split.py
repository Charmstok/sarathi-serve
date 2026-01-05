from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sarathi.time_balance.config import SelectStatsBucketSplitConfig


def _bucket_id(scheduled_tokens: int, *, cfg: SelectStatsBucketSplitConfig) -> str:
    if scheduled_tokens == cfg.chunk_size:
        return f"eq_{cfg.chunk_size}"
    if scheduled_tokens > cfg.chunk_size:
        return f"gt_{cfg.chunk_size}"
    start = (scheduled_tokens // cfg.bucket_size) * cfg.bucket_size
    end = min(start + cfg.bucket_size, cfg.chunk_size - 1)
    return f"{start:03d}_{end:03d}"


def _read_rows(input_csv: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in CSV: {input_csv}")
        rows: List[Dict[str, str]] = []
        for row in reader:
            if row.get("decode_tokens") == "decode_tokens":
                continue
            if not row or all(v is None or str(v).strip() == "" for v in row.values()):
                continue
            rows.append(row)
        return list(reader.fieldnames), rows


def _write_rows(output_csv: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_bucketed_splits(
    *,
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    cfg: SelectStatsBucketSplitConfig,
) -> None:
    fieldnames, rows = _read_rows(input_csv)
    required = {"decode_tokens", "prefill_tokens"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(
            f"CSV missing columns {sorted(missing)}: {input_csv}. Got columns: {fieldnames}"
        )

    rng = random.Random(cfg.seed)

    buckets: Dict[str, List[Dict[str, str]]] = {}
    full_bucket_key = f"eq_{cfg.chunk_size}"
    non_full_total = 0

    for row in rows:
        scheduled = int(float(row["decode_tokens"]) + float(row["prefill_tokens"]))
        key = _bucket_id(scheduled, cfg=cfg)
        buckets.setdefault(key, []).append(row)
        if key != full_bucket_key:
            non_full_total += 1

    # 若某类数据占比过高，可进行降采样处理
    if full_bucket_key in buckets:
        max_full = int(cfg.full_keep_multiplier * non_full_total)
        if max_full >= 0 and len(buckets[full_bucket_key]) > max_full:
            rng.shuffle(buckets[full_bucket_key])
            buckets[full_bucket_key] = buckets[full_bucket_key][:max_full]

    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    # 按桶分层拆分
    for key in sorted(buckets.keys()):
        bucket_rows = buckets[key]
        rng.shuffle(bucket_rows)

        n = len(bucket_rows)
        n_train = int(n * cfg.train_ratio)
        n_val = int(n * cfg.val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train_rows.extend(bucket_rows[:n_train])
        val_rows.extend(bucket_rows[n_train : n_train + n_val])
        test_rows.extend(bucket_rows[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    _write_rows(train_csv, fieldnames, train_rows)
    _write_rows(val_csv, fieldnames, val_rows)
    _write_rows(test_csv, fieldnames, test_rows)


def ensure_bucketed_splits(
    *,
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    cfg: SelectStatsBucketSplitConfig,
) -> None:
    input_path = Path(input_csv)
    outputs = [Path(train_csv), Path(val_csv), Path(test_csv)]

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    def needs_regen() -> bool:
        if any(not p.exists() for p in outputs):
            return True
        input_mtime = input_path.stat().st_mtime
        return any(p.stat().st_mtime < input_mtime for p in outputs)

    if needs_regen():
        create_bucketed_splits(
            input_csv=input_csv,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            cfg=cfg,
        )

