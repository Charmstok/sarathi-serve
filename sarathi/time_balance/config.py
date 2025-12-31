from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# 训练用（同时也是 F() 默认使用的）select stats CSV 路径。
CSV_PATH = (
    "offline_inference_output/2025-12-30_12-32-29/replica_0/select_stats_rank0.csv"
)

# 训练好的 MLP 预测器保存/加载路径。
MODEL_CACHE_PATH = str(Path(__file__).resolve().parent / "time_predictor_mlp.pt")

# 是否在 predict_time.py 中自动加载/保存 MODEL_CACHE_PATH。
ENABLE_MODEL_CACHE = True


@dataclass
class TimePredictorTrainConfig:
    seed: int = field(
        default=0,
        metadata={"help": "训练随机种子。"}
    )
    epochs: int = field(
        default=300,
        metadata={"help": "训练轮数（epochs）。"}
    )
    lr: float = field(
        default=2e-3,
        metadata={"help": "AdamW 学习率。"}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "AdamW 权重衰减（weight_decay）。"}
    )
    hidden_sizes: Tuple[int, ...] = field(
        default=(64, 32),
        metadata={"help": "MLP 回归器隐藏层大小。"},
    )
    batch_size: int = field(
        default=256,
        metadata={"help": "小批量大小（batch size）。"}
    )
    device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch 设备字符串（如 'cuda'/'cpu'）；为 None 时自动选择。"
        },
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "是否打印训练日志。"}
    )
    log_every: int = field(
        default=1,
        metadata={"help": "在 verbose=True 时，每 N 个 epoch 打印一次日志。"}
    )

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
