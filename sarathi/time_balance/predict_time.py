from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from sarathi.time_balance.config import (
    BUCKET_SPLIT_CONFIG,
    CSV_PATH,
    ENABLE_MODEL_CACHE,
    MODEL_CACHE_PATH,
    TEST_CSV_PATH,
    TimePredictorTrainConfig,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
)
from sarathi.time_balance.bucket_split import _bucket_id, ensure_bucketed_splits
INPUT_DIM = 10


BatchState = Mapping[str, Union[int, float, torch.Tensor]]


def _read_select_stats_csv(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Please update `CSV_PATH` in "
    "sarathi/time_balance/config.py to point at your `select_stats_rank*.csv`."
        )

    decode_tokens = []
    sum_decode_context_len = []
    batch_request_count = []
    prefill_tokens = []
    prefill_processed_tokens = []
    gpu_mem_used_mb = []
    gpu_mem_free_mb = []
    cuda_allocated_mb = []
    cuda_reserved_mb = []
    latency_ms = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "decode_tokens",
            "sum_decode_context_len",
            "batch_request_count",
            "prefill_tokens",
            "prefill_processed_tokens",
            "latency_ms",
        }
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"CSV missing columns {sorted(missing)}: {csv_path}. "
                f"Got columns: {reader.fieldnames}"
            )

        has_gpu_mem_used_mb = "gpu_mem_used_mb" in fieldnames
        has_gpu_mem_free_mb = "gpu_mem_free_mb" in fieldnames
        has_cuda_allocated_mb = "cuda_allocated_mb" in fieldnames
        has_cuda_reserved_mb = "cuda_reserved_mb" in fieldnames

        for row in reader:
            # Some CSVs may accidentally contain the header row again (e.g. concatenation).
            if row.get("decode_tokens") == "decode_tokens":
                continue
            if not row or all(v is None or str(v).strip() == "" for v in row.values()):
                continue
            try:
                decode_tokens.append(float(row["decode_tokens"]))
                sum_decode_context_len.append(float(row["sum_decode_context_len"]))
                batch_request_count.append(float(row["batch_request_count"]))
                prefill_tokens.append(float(row["prefill_tokens"]))
                prefill_processed_tokens.append(float(row["prefill_processed_tokens"]))
                gpu_mem_used_mb.append(
                    float(row["gpu_mem_used_mb"]) if has_gpu_mem_used_mb else 0.0
                )
                gpu_mem_free_mb.append(
                    float(row["gpu_mem_free_mb"]) if has_gpu_mem_free_mb else 0.0
                )
                cuda_allocated_mb.append(
                    float(row["cuda_allocated_mb"]) if has_cuda_allocated_mb else 0.0
                )
                cuda_reserved_mb.append(
                    float(row["cuda_reserved_mb"]) if has_cuda_reserved_mb else 0.0
                )
                latency_ms.append(float(row["latency_ms"]))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Bad numeric row in {csv_path} at line {reader.line_num}: {row}"
                ) from e

    x_decode_tokens = torch.tensor(decode_tokens, dtype=torch.float64)
    x_sum_decode_context_len = torch.tensor(sum_decode_context_len, dtype=torch.float64)
    x_batch_request_count = torch.tensor(batch_request_count, dtype=torch.float64)
    x_prefill = torch.tensor(prefill_tokens, dtype=torch.float64)
    x_prefill_processed = torch.tensor(prefill_processed_tokens, dtype=torch.float64)
    x_gpu_mem_used_mb = torch.tensor(gpu_mem_used_mb, dtype=torch.float64)
    x_gpu_mem_free_mb = torch.tensor(gpu_mem_free_mb, dtype=torch.float64)
    x_cuda_allocated_mb = torch.tensor(cuda_allocated_mb, dtype=torch.float64)
    x_cuda_reserved_mb = torch.tensor(cuda_reserved_mb, dtype=torch.float64)
    y_latency = torch.tensor(latency_ms, dtype=torch.float64)

    ones = torch.ones_like(x_decode_tokens)

    # 特征列顺序（训练/预测必须保持一致）:
    # [1, sum_decode_context_len, decode_tokens, batch_request_count, prefill_tokens, prefill_processed_tokens,
    #  gpu_mem_used_mb, gpu_mem_free_mb, cuda_allocated_mb, cuda_reserved_mb]
    x = torch.stack(
        [
            ones,
            x_sum_decode_context_len,
            x_decode_tokens,
            x_batch_request_count,
            x_prefill,
            x_prefill_processed,
            x_gpu_mem_used_mb,
            x_gpu_mem_free_mb,
            x_cuda_allocated_mb,
            x_cuda_reserved_mb,
        ],
        dim=1,
    )

    return x, y_latency


class _MLPRegressor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Iterable[int] = (64, 32),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class TimePredictor:
    model: nn.Module
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    hidden_sizes: Tuple[int, ...]

    @staticmethod
    def from_train_val_csv(
        train_csv_path: str,
        val_csv_path: str,
        *,
        train_config: Optional[TimePredictorTrainConfig] = None,
    ) -> "TimePredictor":
        cfg = train_config or TimePredictorTrainConfig()
        seed = cfg.seed
        epochs = cfg.epochs
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        hidden_sizes: Iterable[int] = cfg.hidden_sizes
        batch_size = cfg.batch_size
        verbose = cfg.verbose
        log_every = cfg.log_every

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x_train, y_train = _read_select_stats_csv(train_csv_path)
        x_val, y_val = _read_select_stats_csv(val_csv_path)
        x_train = x_train.to(dtype=torch.float32)
        y_train = y_train.to(dtype=torch.float32)
        x_val = x_val.to(dtype=torch.float32)
        y_val = y_val.to(dtype=torch.float32)

        # 计算按 scheduled_tokens 分桶的样本权重（稀有桶权重大）。
        w_train: Optional[torch.Tensor] = None
        bucket: Optional[torch.Tensor] = None
        if cfg.use_bucket_weights:
            # feature 顺序: [1, sum_decode_context_len, decode_tokens, batch_request_count, prefill_tokens, ...]
            decode = x_train[:, 2]
            prefill = x_train[:, 4]
            scheduled = torch.round(decode + prefill).to(torch.int64).clamp_min(0)

            chunk_size = int(BUCKET_SPLIT_CONFIG.chunk_size)
            bucket_size = int(BUCKET_SPLIT_CONFIG.bucket_size)
            num_non_full = (chunk_size - 1) // bucket_size + 1
            idx_eq = num_non_full
            idx_gt = num_non_full + 1

            bucket = torch.empty_like(scheduled)
            mask_eq = scheduled == chunk_size
            mask_gt = scheduled > chunk_size
            mask_lt = ~mask_eq & ~mask_gt
            bucket[mask_eq] = idx_eq
            bucket[mask_gt] = idx_gt
            bucket[mask_lt] = torch.div(
                scheduled[mask_lt], bucket_size, rounding_mode="floor"
            )

            counts = torch.bincount(bucket, minlength=idx_gt + 1).to(dtype=torch.float32)
            w_train = 1.0 / counts[bucket].clamp_min(1.0)
            if cfg.bucket_weight_power != 1.0:
                w_train = w_train.pow(float(cfg.bucket_weight_power))
            # 归一化到均值约 1，再裁剪避免极端值。
            w_train = w_train * (w_train.numel() / w_train.sum().clamp_min(1e-12))
            w_train = w_train.clamp(
                min=float(cfg.min_sample_weight), max=float(cfg.max_sample_weight)
            )

        if cfg.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(cfg.device)
        if verbose:
            print(f"[TimePredictor] device={device}, epochs={epochs}, lr={lr}")
            if w_train is not None:
                # 汇总每个分桶的归一化权重
                bucket_weights: dict[str, float] = {}
                unique_buckets = bucket.unique(sorted=True)
                for b in unique_buckets:
                    b_int = int(b.item())
                    # 重建桶标签
                    if b_int == idx_eq:
                        label = f"eq_{chunk_size}"
                    elif b_int == idx_gt:
                        label = f"gt_{chunk_size}"
                    else:
                        start = b_int * bucket_size
                        end = min(start + bucket_size - 1, chunk_size - 1)
                        label = f"{start:03d}_{end:03d}"
                    w_mask = (bucket == b)
                    if w_mask.any():
                        bucket_weights[label] = float(w_train[w_mask][0].item())
                bw_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(bucket_weights.items()))
                print(
                    f"[TimePredictor] weighted_train=on power={cfg.bucket_weight_power} "
                    f"min_w={cfg.min_sample_weight} max_w={cfg.max_sample_weight} "
                    f"bucket_w={{{ {bw_str} }}}"
                )

        # Standardize features/targets for stable MLP training.
        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        y_mean = y_train.mean()
        y_std = y_train.std().clamp_min(1e-6)

        # Keep the explicit intercept feature as a constant 1 after normalization.
        # Otherwise it becomes all-zeros due to std==0 and loses its meaning.
        x_mean[:, 0] = 0.0
        x_std[:, 0] = 1.0

        x_train = (x_train - x_mean) / x_std
        y_train = (y_train - y_mean) / y_std
        x_val = (x_val - x_mean) / x_std
        y_val = (y_val - y_mean) / y_std
        if verbose:
            print(
                f"[TimePredictor] train={x_train.shape[0]}, val={x_val.shape[0]}"
            )

        hidden_sizes_tuple = tuple(int(h) for h in hidden_sizes)
        model = _MLPRegressor(
            in_features=x_train.shape[1], hidden_sizes=hidden_sizes_tuple
        ).to(
            device=device
        )
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_mae: float = float("inf")

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            # Simple mini-batching without DataLoader to keep deps minimal.
            order = torch.randperm(x_train.shape[0])
            x_train_shuf = x_train[order]
            y_train_shuf = y_train[order]
            w_train_shuf = w_train[order] if w_train is not None else None

            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, x_train_shuf.shape[0], max(1, batch_size)):
                end = min(start + batch_size, x_train.shape[0])
                xb = x_train_shuf[start:end].to(device=device)
                yb = y_train_shuf[start:end].to(device=device)

                pred = model(xb)
                if w_train_shuf is None:
                    loss = ((pred - yb) ** 2).mean()
                else:
                    wb = w_train_shuf[start:end].to(device=device)
                    loss = (((pred - yb) ** 2) * wb).mean()
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu().item())
                num_batches += 1

            model.eval()
            with torch.no_grad():
                pred_val = model(x_val.to(device=device))
                val_mae = (pred_val - y_val.to(device=device)).abs().mean().item()
                val_mae_ms = val_mae * float(y_std.item())
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

            if verbose and (epoch % max(1, log_every) == 0):
                avg_loss = epoch_loss / max(1, num_batches)
                print(
                    f"[TimePredictor] epoch={epoch}/{epochs} train_mse={avg_loss:.6f} "
                    f"val_mae(norm)={val_mae:.6f} val_mae_ms={val_mae_ms:.3f} best={best_val_mae:.6f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        return TimePredictor(
            model=model.to(device=torch.device("cpu")),
            x_mean=x_mean.squeeze(0).to(dtype=torch.float32, device="cpu"),
            x_std=x_std.squeeze(0).to(dtype=torch.float32, device="cpu"),
            y_mean=y_mean.to(dtype=torch.float32, device="cpu"),
            y_std=y_std.to(dtype=torch.float32, device="cpu"),
            hidden_sizes=hidden_sizes_tuple,
        )

    def save(self, path: str) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "hidden_sizes": self.hidden_sizes,
            "in_features": INPUT_DIM,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load(path: str) -> "TimePredictor":
        payload = torch.load(path, map_location="cpu")
        hidden_sizes = tuple(int(h) for h in payload["hidden_sizes"])
        in_features = int(payload.get("in_features", INPUT_DIM))
        if in_features != INPUT_DIM:
            raise ValueError(
                f"Incompatible cached model: in_features={in_features}, expected {INPUT_DIM}. "
                f"Delete {path} to retrain."
            )
        model = _MLPRegressor(in_features=in_features, hidden_sizes=hidden_sizes)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return TimePredictor(
            model=model,
            x_mean=payload["x_mean"].to(dtype=torch.float32, device="cpu"),
            x_std=payload["x_std"].to(dtype=torch.float32, device="cpu"),
            y_mean=payload["y_mean"].to(dtype=torch.float32, device="cpu"),
            y_std=payload["y_std"].to(dtype=torch.float32, device="cpu"),
            hidden_sizes=hidden_sizes,
        )

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(
                f"Expected features[..., {INPUT_DIM}] with columns "
                "[1, sum_decode_context_len, decode, batch_request_count, prefill, prefill_processed, "
                "gpu_mem_used_mb, gpu_mem_free_mb, cuda_allocated_mb, cuda_reserved_mb]. "
                f"Got shape {tuple(features.shape)}"
            )
        x = features.to(dtype=torch.float32)
        x_mean = self.x_mean.to(device=x.device, dtype=x.dtype)
        x_std = self.x_std.to(device=x.device, dtype=x.dtype)
        x_norm = (x - x_mean) / x_std

        # Keep model on CPU; this is a lightweight predictor.
        with torch.no_grad():
            y_norm = self.model(x_norm.cpu()).to(device=x.device, dtype=x.dtype)
        y_mean = self.y_mean.to(device=x.device, dtype=x.dtype)
        y_std = self.y_std.to(device=x.device, dtype=x.dtype)
        return y_norm * y_std + y_mean

    def predict(
        self,
        *,
        decode_tokens: Union[int, float, torch.Tensor],
        sum_decode_context_len: Union[int, float, torch.Tensor],
        batch_request_count: Union[int, float, torch.Tensor],
        prefill_tokens: Union[int, float, torch.Tensor],
        prefill_processed_tokens: Union[int, float, torch.Tensor],
        gpu_mem_used_mb: Union[int, float, torch.Tensor, None] = None,
        gpu_mem_free_mb: Union[int, float, torch.Tensor, None] = None,
        cuda_allocated_mb: Union[int, float, torch.Tensor, None] = None,
        cuda_reserved_mb: Union[int, float, torch.Tensor, None] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if (
            gpu_mem_used_mb is None
            or gpu_mem_free_mb is None
            or cuda_allocated_mb is None
            or cuda_reserved_mb is None
        ):
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                    used_mb = float(total_bytes - free_bytes) / (1024.0 * 1024.0)
                    free_mb = float(free_bytes) / (1024.0 * 1024.0)
                    alloc_mb = float(torch.cuda.memory_allocated(device_idx)) / (
                        1024.0 * 1024.0
                    )
                    reserv_mb = float(torch.cuda.memory_reserved(device_idx)) / (
                        1024.0 * 1024.0
                    )
                except Exception:
                    used_mb, free_mb, alloc_mb, reserv_mb = 0.0, 0.0, 0.0, 0.0
            else:
                used_mb, free_mb, alloc_mb, reserv_mb = 0.0, 0.0, 0.0, 0.0

            if gpu_mem_used_mb is None:
                gpu_mem_used_mb = used_mb
            if gpu_mem_free_mb is None:
                gpu_mem_free_mb = free_mb
            if cuda_allocated_mb is None:
                cuda_allocated_mb = alloc_mb
            if cuda_reserved_mb is None:
                cuda_reserved_mb = reserv_mb

        dt = torch.as_tensor(decode_tokens, dtype=dtype, device=device)
        sdcl = torch.as_tensor(sum_decode_context_len, dtype=dtype, device=device)
        brc = torch.as_tensor(batch_request_count, dtype=dtype, device=device)
        pt = torch.as_tensor(prefill_tokens, dtype=dtype, device=device)
        ppt = torch.as_tensor(prefill_processed_tokens, dtype=dtype, device=device)
        gmem_used = torch.as_tensor(gpu_mem_used_mb, dtype=dtype, device=device)
        gmem_free = torch.as_tensor(gpu_mem_free_mb, dtype=dtype, device=device)
        cuda_alloc = torch.as_tensor(cuda_allocated_mb, dtype=dtype, device=device)
        cuda_resv = torch.as_tensor(cuda_reserved_mb, dtype=dtype, device=device)
        dt, sdcl, brc, pt, ppt, gmem_used, gmem_free, cuda_alloc, cuda_resv = (
            torch.broadcast_tensors(
                dt, sdcl, brc, pt, ppt, gmem_used, gmem_free, cuda_alloc, cuda_resv
            )
        )
        ones = torch.ones_like(dt, dtype=dtype, device=dt.device)
        features = torch.stack(
            [
                ones,
                sdcl,
                dt,
                brc,
                pt,
                ppt,
                gmem_used,
                gmem_free,
                cuda_alloc,
                cuda_resv,
            ],
            dim=-1,
        )
        return self.predict_from_features(features)


def _eval_mae(predictor: TimePredictor, csv_path: str) -> float:
    x, y = _read_select_stats_csv(csv_path)
    yhat = predictor.predict_from_features(x.to(dtype=torch.float32))
    return (yhat - y.to(dtype=torch.float32)).abs().mean().item()


if __name__ == "__main__":
    ensure_bucketed_splits(
        input_csv=CSV_PATH,
        train_csv=TRAIN_CSV_PATH,
        val_csv=VAL_CSV_PATH,
        test_csv=TEST_CSV_PATH,
        cfg=BUCKET_SPLIT_CONFIG,
    )

    predictor = TimePredictor.from_train_val_csv(
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        train_config=TimePredictorTrainConfig(verbose=True, log_every=1),
    )
    if ENABLE_MODEL_CACHE:
        if os.path.exists(MODEL_CACHE_PATH):
            os.remove(MODEL_CACHE_PATH)
        predictor.save(MODEL_CACHE_PATH)
        print(f"[TimePredictor] saved model to {MODEL_CACHE_PATH}")

    train_mae = _eval_mae(predictor, TRAIN_CSV_PATH)
    val_mae = _eval_mae(predictor, VAL_CSV_PATH)
    test_mae = _eval_mae(predictor, TEST_CSV_PATH)
    print("[TimePredictor] train_mae_ms =", train_mae)
    print("[TimePredictor] val_mae_ms   =", val_mae)
    print("[TimePredictor] test_mae_ms  =", test_mae)
