from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchRuntimeStats:
    """Runtime stats collected on the worker during batch execution."""

    gpu_mem_used_mb: float
    gpu_mem_free_mb: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float
