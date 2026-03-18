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

INPUT_DIM = 16


BatchState = Mapping[str, Union[int, float, torch.Tensor]]


def _build_feature_tensor(
    *,
    decode_tokens: torch.Tensor,
    sum_decode_context_len: torch.Tensor,
    batch_request_count: torch.Tensor,
    prefill_tokens: torch.Tensor,
    prefill_processed_tokens: torch.Tensor,
    max_decode_context_len: torch.Tensor,
    max_prefill_processed_tokens: torch.Tensor,
    gpu_mem_used_mb: torch.Tensor,
    gpu_mem_free_mb: torch.Tensor,
    cuda_allocated_mb: torch.Tensor,
    cuda_reserved_mb: torch.Tensor,
) -> torch.Tensor:
    """将原始调度统计量拼接并扩展成模型输入的 16 维特征张量。"""

    # 把原始调度统计量扩展成模型实际使用的 16 维特征。
    # 这里除了直接输入原始字段，还额外构造了一些交叉项，帮助 MLP 更容易拟合延迟模式。
    scheduled_tokens = decode_tokens + prefill_tokens
    avg_decode_ctx = sum_decode_context_len / decode_tokens.clamp_min(1.0)
    decode_ctx_interaction = decode_tokens * avg_decode_ctx
    prefill_interaction = prefill_tokens * prefill_processed_tokens
    ones = torch.ones_like(decode_tokens)

    return torch.stack(
        [
            ones,
            sum_decode_context_len,
            decode_tokens,
            batch_request_count,
            prefill_tokens,
            prefill_processed_tokens,
            scheduled_tokens,
            avg_decode_ctx,
            decode_ctx_interaction,
            prefill_interaction,
            max_decode_context_len,
            max_prefill_processed_tokens,
            gpu_mem_used_mb,
            gpu_mem_free_mb,
            cuda_allocated_mb,
            cuda_reserved_mb,
        ],
        dim=-1,
    )


def _weighted_asymmetric_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float,
    underpredict_weight: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算带样本权重、且对低估更敏感的 Huber 回归损失。"""
    
    # 使用非对称 Huber loss：
    # 1. Huber 比纯 MSE 对异常值更稳健；
    # 2. 当预测值低于真实值时，施加更大的惩罚，降低“低估延迟”的风险；
    # 3. 可叠加样本权重，让稀有桶样本在训练时更重要。
    error = pred - target
    abs_error = error.abs()
    delta_tensor = torch.as_tensor(delta, dtype=pred.dtype, device=pred.device)
    huber = torch.where(
        abs_error <= delta_tensor,
        0.5 * error.square(),
        delta_tensor * (abs_error - 0.5 * delta_tensor),
    )
    asym_weight = torch.where(
        error < 0,
        torch.as_tensor(
            underpredict_weight,
            dtype=pred.dtype,
            device=pred.device,
        ),
        torch.ones_like(error),
    )
    loss = huber * asym_weight
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def _read_select_stats_csv(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """从训练/验证 CSV 读取样本，并返回特征张量和延迟标签。"""
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
    max_decode_context_len = []
    max_prefill_processed_tokens = []
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
            "max_decode_context_len",
            "max_prefill_processed_tokens",
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
            # 兼容被拼接过的 CSV：中间若混入重复表头或空行，直接跳过。
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
                max_decode_context_len.append(float(row["max_decode_context_len"]))
                max_prefill_processed_tokens.append(
                    float(row["max_prefill_processed_tokens"])
                )
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
    x_max_decode_context_len = torch.tensor(
        max_decode_context_len, dtype=torch.float64
    )
    x_max_prefill_processed_tokens = torch.tensor(
        max_prefill_processed_tokens, dtype=torch.float64
    )
    x_gpu_mem_used_mb = torch.tensor(gpu_mem_used_mb, dtype=torch.float64)
    x_gpu_mem_free_mb = torch.tensor(gpu_mem_free_mb, dtype=torch.float64)
    x_cuda_allocated_mb = torch.tensor(cuda_allocated_mb, dtype=torch.float64)
    x_cuda_reserved_mb = torch.tensor(cuda_reserved_mb, dtype=torch.float64)
    y_latency = torch.tensor(latency_ms, dtype=torch.float64)

    x = _build_feature_tensor(
        decode_tokens=x_decode_tokens,
        sum_decode_context_len=x_sum_decode_context_len,
        batch_request_count=x_batch_request_count,
        prefill_tokens=x_prefill,
        prefill_processed_tokens=x_prefill_processed,
        max_decode_context_len=x_max_decode_context_len,
        max_prefill_processed_tokens=x_max_prefill_processed_tokens,
        gpu_mem_used_mb=x_gpu_mem_used_mb,
        gpu_mem_free_mb=x_gpu_mem_free_mb,
        cuda_allocated_mb=x_cuda_allocated_mb,
        cuda_reserved_mb=x_cuda_reserved_mb,
    )

    # x 是模型输入特征，y 是监督信号 latency_ms。
    return x, y_latency


class _MLPRegressor(nn.Module):
    """用于延迟回归的轻量级多层感知机。"""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Iterable[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        """按给定输入维度、隐藏层配置和 dropout 构建 MLP。"""
        super().__init__()
        # 一个标准的多层感知机回归器：
        # Linear -> ReLU -> (Dropout) -> ... -> Linear(1)
        layers: list[nn.Module] = []
        prev = in_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers) # “*”解包 --- 把列表解包成多个独立参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对一批特征做前向传播，返回一维回归结果。"""
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class TimePredictor:
    """封装训练好的延迟预测模型及其标准化参数。"""

    model: nn.Module
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor
    hidden_sizes: Tuple[int, ...]
    dropout: float = 0.0

    @staticmethod
    def from_train_val_csv(
        train_csv_path: str,
        val_csv_path: str,
        *,
        train_config: Optional[TimePredictorTrainConfig] = None,
    ) -> "TimePredictor":
        """从训练集和验证集 CSV 训练模型，并返回可直接推理的预测器。"""
        # 这个静态方法就是完整的训练入口：
        # 读数据 -> 标准化 -> 构建模型 -> 训练 -> 选最佳验证集权重 -> 返回 predictor。
        cfg = train_config or TimePredictorTrainConfig()
        seed = cfg.seed
        epochs = cfg.epochs
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        hidden_sizes: Iterable[int] = cfg.hidden_sizes
        dropout = float(cfg.dropout)
        batch_size = cfg.batch_size
        huber_delta = float(cfg.huber_delta)
        underpredict_weight = float(cfg.underpredict_weight)
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

        # 按 scheduled_tokens 分桶，并给稀有桶更高权重。
        # 这样可以避免训练被高频桶主导，提升长尾场景的拟合能力。
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
            print(
                f"[TimePredictor] device={device}, epochs={epochs}, lr={lr}, "
                f"huber_delta={huber_delta}, underpredict_weight={underpredict_weight}"
            )
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

        # 对输入和标签做标准化，减小不同量纲的影响，让 MLP 更容易训练稳定。
        x_mean = x_train.mean(dim=0, keepdim=True)
        x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        y_mean = y_train.mean()
        y_std = y_train.std().clamp_min(1e-6)

        # 第一列是显式常数项（intercept feature），标准化后仍保留为常数 1。
        # 否则它会因为方差为 0 被压成全 0，失去“偏置项”的作用。
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
            in_features=x_train.shape[1],
            hidden_sizes=hidden_sizes_tuple,
            dropout=dropout,
        ).to(
            device=device
        )
        # AdamW 用于优化 MLP 参数；weight_decay 相当于 L2 正则的一种实现。
        optim = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_val_mae: float = float("inf")

        # 训练模型
        for epoch in range(1, max(1, epochs) + 1):
            model.train()
            # 训练时每个 epoch 打乱样本顺序，然后手工做 mini-batch。
            # 这里没有引入 DataLoader，逻辑更直接，依赖也更少。
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

                # 标准的 PyTorch 训练闭环：
                # 前向计算 -> loss -> 清梯度 -> 反向传播 -> 参数更新。
                pred = model(xb)
                wb = (
                    w_train_shuf[start:end].to(device=device)
                    if w_train_shuf is not None
                    else None
                )
                loss = _weighted_asymmetric_huber_loss(
                    pred,
                    yb,
                    delta=huber_delta,
                    underpredict_weight=underpredict_weight,
                    sample_weight=wb,
                )
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.detach().cpu().item())
                num_batches += 1

            model.eval()
            with torch.no_grad():
                pred_val = model(x_val.to(device=device))
                # 验证集用 MAE 挑最优模型；最终保留验证误差最低的那一版权重。
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
                    f"[TimePredictor] epoch={epoch}/{epochs} train_loss={avg_loss:.6f} "
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
            dropout=dropout,
        )

    def save(self, path: str) -> None:
        """把模型权重和标准化统计量一起保存到磁盘。"""
        # 除了模型权重，也一起保存训练时的标准化统计量。
        # 这样推理时才能复用同一套归一化方式。
        payload = {
            "state_dict": self.model.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "hidden_sizes": self.hidden_sizes,
            "dropout": float(getattr(self, "dropout", 0.0)),
            "in_features": INPUT_DIM,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)

    @staticmethod
    def load(path: str) -> "TimePredictor":
        """从磁盘恢复预测器及其对应的归一化配置。"""
        payload = torch.load(path, map_location="cpu")
        hidden_sizes = tuple(int(h) for h in payload["hidden_sizes"])
        in_features = int(payload.get("in_features", INPUT_DIM))
        if "dropout" not in payload:
            raise ValueError(
                f"Incompatible cached model: missing 'dropout' field in {path}. "
                f"Delete {path} to retrain."
            )
        dropout = float(payload["dropout"])
        if in_features != INPUT_DIM:
            raise ValueError(
                f"Incompatible cached model: in_features={in_features}, expected {INPUT_DIM}. "
                f"Delete {path} to retrain."
            )
        model = _MLPRegressor(
            in_features=in_features, hidden_sizes=hidden_sizes, dropout=dropout
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return TimePredictor(
            model=model,
            x_mean=payload["x_mean"].to(dtype=torch.float32, device="cpu"),
            x_std=payload["x_std"].to(dtype=torch.float32, device="cpu"),
            y_mean=payload["y_mean"].to(dtype=torch.float32, device="cpu"),
            y_std=payload["y_std"].to(dtype=torch.float32, device="cpu"),
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """对已经构造好的特征张量做推理，返回毫秒级延迟预测。"""
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(
                f"Expected features[..., {INPUT_DIM}] with columns "
                "[1, sum_decode_context_len, decode_tokens, batch_request_count, "
                "prefill_tokens, prefill_processed_tokens, scheduled_tokens, "
                "avg_decode_ctx, decode_ctx_interaction, prefill_interaction, "
                "max_decode_context_len, max_prefill_processed_tokens, "
                "gpu_mem_used_mb, gpu_mem_free_mb, cuda_allocated_mb, "
                "cuda_reserved_mb]. "
                f"Got shape {tuple(features.shape)}"
            )
        x = features.to(dtype=torch.float32)
        x_mean = self.x_mean.to(device=x.device, dtype=x.dtype)
        x_std = self.x_std.to(device=x.device, dtype=x.dtype)
        x_norm = (x - x_mean) / x_std

        # 推理阶段先按训练时统计量做标准化，再把预测值反标准化回毫秒。
        # 模型本身保持在 CPU 上即可，开销很小。
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
        max_decode_context_len: Union[int, float, torch.Tensor],
        max_prefill_processed_tokens: Union[int, float, torch.Tensor],
        gpu_mem_used_mb: Union[int, float, torch.Tensor, None] = None,
        gpu_mem_free_mb: Union[int, float, torch.Tensor, None] = None,
        cuda_allocated_mb: Union[int, float, torch.Tensor, None] = None,
        cuda_reserved_mb: Union[int, float, torch.Tensor, None] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """接收原始调度统计量，自动构造特征后执行延迟预测。"""
        # 这是对外更方便的推理接口：
        # 传入原始调度统计量，内部自动补齐显存信息、拼装特征并调用模型。
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
        mdcl = torch.as_tensor(max_decode_context_len, dtype=dtype, device=device)
        mppt = torch.as_tensor(
            max_prefill_processed_tokens, dtype=dtype, device=device
        )
        gmem_used = torch.as_tensor(gpu_mem_used_mb, dtype=dtype, device=device)
        gmem_free = torch.as_tensor(gpu_mem_free_mb, dtype=dtype, device=device)
        cuda_alloc = torch.as_tensor(cuda_allocated_mb, dtype=dtype, device=device)
        cuda_resv = torch.as_tensor(cuda_reserved_mb, dtype=dtype, device=device)
        (
            dt,
            sdcl,
            brc,
            pt,
            ppt,
            mdcl,
            mppt,
            gmem_used,
            gmem_free,
            cuda_alloc,
            cuda_resv,
        ) = torch.broadcast_tensors(
            dt,
            sdcl,
            brc,
            pt,
            ppt,
            mdcl,
            mppt,
            gmem_used,
            gmem_free,
            cuda_alloc,
            cuda_resv,
        )
        features = _build_feature_tensor(
            decode_tokens=dt,
            sum_decode_context_len=sdcl,
            batch_request_count=brc,
            prefill_tokens=pt,
            prefill_processed_tokens=ppt,
            max_decode_context_len=mdcl,
            max_prefill_processed_tokens=mppt,
            gpu_mem_used_mb=gmem_used,
            gpu_mem_free_mb=gmem_free,
            cuda_allocated_mb=cuda_alloc,
            cuda_reserved_mb=cuda_resv,
        )
        return self.predict_from_features(features)


def _eval_mae(predictor: TimePredictor, csv_path: str) -> float:
    """在给定 CSV 数据集上评估预测器的平均绝对误差。"""
    # 评估给定 CSV 上的平均绝对误差，单位仍然是毫秒。
    x, y = _read_select_stats_csv(csv_path)
    yhat = predictor.predict_from_features(x.to(dtype=torch.float32))
    return (yhat - y.to(dtype=torch.float32)).abs().mean().item()


if __name__ == "__main__":
    # 直接执行本文件时，会自动完成：
    # 1. 按桶切分 train/val/test
    # 2. 训练 MLP 预测器
    # 3. 保存模型
    # 4. 打印三份数据集上的 MAE
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
