from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List


class ChunkSearch(ABC):
    @abstractmethod
    def min_score(
        self,
        low: int,
        high: int,
        score_fn: Callable[[int], float],
    ) -> int:
        """Return a chunk size in [low, high] minimizing a scalar score."""


class GridChunkSearch(ChunkSearch):
    def __init__(self, *, step: int = 32) -> None:
        if step <= 0:
            raise ValueError("step must be > 0.")
        self.step = int(step)

    def _build_candidates(self, low: int, high: int) -> List[int]:
        candidates = {low, high}
        step = self.step
        first = ((low + step - 1) // step) * step
        for x in range(first, high + 1, step):
            candidates.add(x)
        return sorted(candidates)

    def min_score(
        self,
        low: int,
        high: int,
        score_fn: Callable[[int], float],
    ) -> int:
        if high < low:
            return low - 1

        best_x = low
        best_score = float("inf")
        for x in self._build_candidates(low, high):
            if x < low or x > high:
                continue
            score = float(score_fn(x))
            if score < best_score or (score == best_score and x > best_x):
                best_score = score
                best_x = x
        return best_x


class ChunkSearchFactory:
    _SEARCH_REGISTRY = {
        "grid": GridChunkSearch,
    }

    @classmethod
    def get_search(cls, search_name: str, **kwargs) -> ChunkSearch:
        return cls._SEARCH_REGISTRY[search_name](**kwargs)

    @classmethod
    def get_available_searches(cls) -> List[str]:
        return list(cls._SEARCH_REGISTRY.keys())
