from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sarathi.core.datatypes.runtime_stats import BatchRuntimeStats
from sarathi.core.datatypes.sequence import SamplerOutputs


@dataclass(frozen=True)
class ModelStepResult:
    sampler_outputs: Optional[SamplerOutputs]
    runtime_stats: BatchRuntimeStats
