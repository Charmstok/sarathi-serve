from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

CSV_PATH = (
    "offline_inference_output/2025-12-26_17-28-15/replica_0/select_stats_rank0.csv"
)
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "time_predictor_mlp.pt")
ENABLE_MODEL_CACHE = True
INPUT_DIM = 6


BatchState = Mapping[str, Union[int, float, torch.Tensor]]


def _read_select_stats_csv(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Please update `CSV_PATH` in "
            f"{__file__} to point at your `select_stats_rank*.csv`."
        )

    decode_tokens = []
    decode_history_tokens = []
    batch_request_count = []
    prefill_tokens = []
    prefill_processed_tokens = []
    latency_ms = []

    with open(csv_path, newline="", encoding="utf-8") as f:
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
                f"CSV missing columns {sorted(missing)}: {csv_path}. "
                f"Got columns: {reader.fieldnames}"
            )

        for row in reader:
            decode_tokens.append(float(row["decode_tokens"]))
            decode_history_tokens.append(float(row["decode_history_tokens"]))
            batch_request_count.append(float(row["batch_request_count"]))
            prefill_tokens.append(float(row["prefill_tokens"]))
            prefill_processed_tokens.append(float(row["prefill_processed_tokens"]))
            latency_ms.append(float(row["latency_ms"]))

    x_decode_tokens = torch.tensor(decode_tokens, dtype=torch.float64)
    x_decode_history = torch.tensor(decode_history_tokens, dtype=torch.float64)
    x_batch_request_count = torch.tensor(batch_request_count, dtype=torch.float64)
    x_prefill = torch.tensor(prefill_tokens, dtype=torch.float64)
    x_prefill_processed = torch.tensor(prefill_processed_tokens, dtype=torch.float64)
    y_latency = torch.tensor(latency_ms, dtype=torch.float64)

    ones = torch.ones_like(x_decode_tokens)

    # 设计矩阵列顺序严格对应数学模型:
    # [1, decode_history_tokens, decode_tokens, batch_request_count, prefill_tokens, prefill_processed_tokens]
    x = torch.stack(
        [
            ones,
            x_decode_history,
            x_decode_tokens,
            x_batch_request_count,
            x_prefill,
            x_prefill_processed,
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
    def from_csv(
        csv_path: str = CSV_PATH,
        *,
        seed: int = 0,
        epochs: int = 300,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        hidden_sizes: Iterable[int] = (64, 32),
        batch_size: int = 256,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        log_every: int = 1,
    ) -> "TimePredictor":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x, y = _read_select_stats_csv(csv_path)
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"[TimePredictor] device={device}, epochs={epochs}, lr={lr}")

        # Standardize features/targets for stable MLP training.
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        y_mean = y.mean()
        y_std = y.std().clamp_min(1e-6)

        # Keep the explicit intercept feature as a constant 1 after normalization.
        # Otherwise it becomes all-zeros due to std==0 and loses its meaning.
        x_mean[:, 0] = 0.0
        x_std[:, 0] = 1.0

        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std

        # Train/val split.
        n = x_norm.shape[0]
        perm = torch.randperm(n)
        split = max(int(n * 0.8), 1)
        train_idx, val_idx = perm[:split], perm[split:]

        x_train = x_norm[train_idx]
        y_train = y_norm[train_idx]
        x_val = x_norm[val_idx] if val_idx.numel() else None
        y_val = y_norm[val_idx] if val_idx.numel() else None
        if verbose:
            print(
                f"[TimePredictor] samples={n}, train={x_train.shape[0]}, val={0 if x_val is None else x_val.shape[0]}"
            )

        hidden_sizes_tuple = tuple(int(h) for h in hidden_sizes)
        model = _MLPRegressor(
            in_features=x.shape[1], hidden_sizes=hidden_sizes_tuple
        ).to(
            device=device
        )
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.MSELoss()

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_mae: float = float("inf")

        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            # Simple mini-batching without DataLoader to keep deps minimal.
            order = torch.randperm(x_train.shape[0])
            x_train_shuf = x_train[order]
            y_train_shuf = y_train[order]

            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, x_train_shuf.shape[0], max(1, batch_size)):
                end = min(start + batch_size, x_train.shape[0])
                xb = x_train_shuf[start:end].to(device=device)
                yb = y_train_shuf[start:end].to(device=device)

                pred = model(xb)
                loss = loss_fn(pred, yb)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu().item())
                num_batches += 1

            val_mae = None
            val_mae_ms = None
            if x_val is not None:
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
                if val_mae is None:
                    print(
                        f"[TimePredictor] epoch={epoch}/{epochs} train_mse={avg_loss:.6f}"
                    )
                else:
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
                "[1, decode_history, decode, batch_request_count, prefill, prefill_processed]. "
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
        decode_history_tokens: Union[int, float, torch.Tensor],
        batch_request_count: Union[int, float, torch.Tensor],
        prefill_tokens: Union[int, float, torch.Tensor],
        prefill_processed_tokens: Union[int, float, torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        dt = torch.as_tensor(decode_tokens, dtype=dtype, device=device)
        dht = torch.as_tensor(decode_history_tokens, dtype=dtype, device=device)
        brc = torch.as_tensor(batch_request_count, dtype=dtype, device=device)
        pt = torch.as_tensor(prefill_tokens, dtype=dtype, device=device)
        ppt = torch.as_tensor(prefill_processed_tokens, dtype=dtype, device=device)
        dt, dht, brc, pt, ppt = torch.broadcast_tensors(dt, dht, brc, pt, ppt)
        ones = torch.ones_like(dt, dtype=dtype, device=dt.device)
        features = torch.stack([ones, dht, dt, brc, pt, ppt], dim=-1)
        return self.predict_from_features(features)


_CACHED: dict[str, TimePredictor] = {}


def _get_predictor(csv_path: str = CSV_PATH) -> TimePredictor:
    predictor = _CACHED.get(csv_path)
    if predictor is None:
        if ENABLE_MODEL_CACHE and os.path.exists(MODEL_CACHE_PATH):
            try:
                predictor = TimePredictor.load(MODEL_CACHE_PATH)
            except Exception:
                predictor = TimePredictor.from_csv(csv_path)
        else:
            predictor = TimePredictor.from_csv(csv_path)
        if ENABLE_MODEL_CACHE:
            try:
                predictor.save(MODEL_CACHE_PATH)
            except Exception:
                pass
        _CACHED[csv_path] = predictor
    return predictor


def F(batch_state: BatchState, *, csv_path: str = CSV_PATH) -> torch.Tensor:
    """
    输入当前 Batch 状态，输出预测执行时间 T_pred (ms)。

    输入特征:
      decode_tokens, decode_history_tokens, batch_request_count,
      prefill_tokens, prefill_processed_tokens
    """
    predictor = _get_predictor(csv_path)

    try:
        decode_tokens = batch_state["decode_tokens"]
        decode_history_tokens = batch_state["decode_history_tokens"]
        batch_request_count = batch_state["batch_request_count"]
        prefill_tokens = batch_state["prefill_tokens"]
        prefill_processed_tokens = batch_state["prefill_processed_tokens"]
    except KeyError as e:
        raise KeyError(
            "batch_state must contain keys: decode_tokens, decode_history_tokens, "
            "batch_request_count, prefill_tokens, prefill_processed_tokens"
        ) from e

    return predictor.predict(
        decode_tokens=decode_tokens,
        decode_history_tokens=decode_history_tokens,
        batch_request_count=batch_request_count,
        prefill_tokens=prefill_tokens,
        prefill_processed_tokens=prefill_processed_tokens,
    )


def _eval_mae(predictor: TimePredictor, csv_path: str) -> float:
    x, y = _read_select_stats_csv(csv_path)
    yhat = predictor.predict_from_features(x.to(dtype=torch.float32))
    return (yhat - y.to(dtype=torch.float32)).abs().mean().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = TimePredictor.from_csv(
        CSV_PATH,
        device=device,
        verbose=True,
        log_every=1,
    )

    mae = _eval_mae(predictor, CSV_PATH)
    print("[TimePredictor] train_mae_ms =", mae)


    # Example usage
    batch = {
        "decode_tokens": 28,
        "decode_history_tokens": 2240,
        "batch_request_count": 32,
        "prefill_tokens": 228,
        "prefill_processed_tokens": 179,
    }
    print("T_pred_ms =", F(batch).item())
