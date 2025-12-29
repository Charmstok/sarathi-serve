from __future__ import annotations

from pathlib import Path

# Path to the select stats CSV used for training (and as default for F()).
CSV_PATH = (
    "offline_inference_output/2025-12-29_14-25-24/replica_0/select_stats_rank0.csv"
)

# Where to save/load the trained MLP predictor.
MODEL_CACHE_PATH = str(Path(__file__).resolve().parent / "time_predictor_mlp.pt")

# Whether to load/save MODEL_CACHE_PATH automatically in predict_time.py.
ENABLE_MODEL_CACHE = True

