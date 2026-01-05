from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# 训练用（同时也是 F() 默认使用的）select stats CSV 路径。
CSV_PATH = (
    "offline_inference_output/2026-01-05_15-51-09/replica_0/select_stats_rank0.csv"
)

# 训练好的 MLP 预测器保存/加载路径。
MODEL_CACHE_PATH = str(Path(__file__).resolve().parent / "time_predictor_mlp_v6.pt")

# 是否在 predict_time.py 中自动加载/保存 MODEL_CACHE_PATH。
ENABLE_MODEL_CACHE = True


@dataclass
class SelectStatsBucketSplitConfig:
    chunk_size: int = field(
        default=256,
        metadata={"help": "调度 chunk_size（用于区分 scheduled_tokens==chunk_size 这一桶）。"},
    )
    bucket_size: int = field(
        default=32,
        metadata={"help": "scheduled_tokens 分桶的步长，例如 32 对应 0-32,33-64,..."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "分桶采样与分层切分的随机种子。"},
    )
    train_ratio: float = field(
        default=0.8,
        metadata={"help": "训练集比例（分桶内分层切分）。"},
    )
    val_ratio: float = field(
        default=0.1,
        metadata={"help": "验证集比例（分桶内分层切分）。"},
    )
    test_ratio: float = field(
        default=0.1,
        metadata={"help": "测试集比例（分桶内分层切分）。"},
    )
    full_keep_multiplier: float = field(
        default=2.0,
        metadata={
            "help": "对 scheduled_tokens==chunk_size 的桶进行下采样：最多保留 full_keep_multiplier * non_full_total 条。"
        },
    )

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio+val_ratio+test_ratio must equal 1.0, got {total}."
            )
        if self.bucket_size <= 0:
            raise ValueError("bucket_size must be > 0.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")
        if self.full_keep_multiplier < 0:
            raise ValueError("full_keep_multiplier must be >= 0.")


BUCKET_SPLIT_CONFIG = SelectStatsBucketSplitConfig()

_raw = Path(CSV_PATH)
TRAIN_CSV_PATH = str(_raw.with_name(_raw.stem + "_train.csv"))
VAL_CSV_PATH = str(_raw.with_name(_raw.stem + "_val.csv"))
TEST_CSV_PATH = str(_raw.with_name(_raw.stem + "_test.csv"))


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
    # 加权训练配置（按 scheduled_tokens 分桶，稀有桶权重大）
    use_bucket_weights: bool = field(
        default=True,
        metadata={
            "help": "是否按 scheduled_tokens 分桶频次做样本加权（稀有桶权重大，常见桶权重小）。"
        },
    )
    bucket_weight_power: float = field(
        default=1.0,
        metadata={
            "help": "桶权重幂次: w = (1/count)^power；power 越大越强调稀有桶。"
        },
    )
    max_sample_weight: float = field(
        default=10.0,
        metadata={"help": "权重裁剪上限（归一化后平均权重约为 1）。"},
    )
    min_sample_weight: float = field(
        default=1.0,
        metadata={"help": "权重裁剪下限，避免主导桶权重过小（归一化后再裁剪）。"},
    )

    def __post_init__(self) -> None:
        self.hidden_sizes = tuple(int(h) for h in self.hidden_sizes)
        if self.bucket_weight_power < 0:
            raise ValueError("bucket_weight_power must be >= 0.")
        if self.max_sample_weight <= 0:
            raise ValueError("max_sample_weight must be > 0.")
        if self.min_sample_weight <= 0:
            raise ValueError("min_sample_weight must be > 0.")
