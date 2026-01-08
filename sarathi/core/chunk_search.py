from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List



class ChunkSearch(ABC):
    @abstractmethod
    def max_true(
        self,
        low: int,
        high: int,
        predicate: Callable[[int], bool],
    ) -> int:
        """Return a chunk size in [low, high] that satisfies predicate.

        Different implementations may have different assumptions:
        - BinaryChunkSearch assumes predicate is monotonic (True...False).
        - GridChunkSearch makes no monotonicity assumption and scans candidates.
        """


class BinaryChunkSearch(ChunkSearch):
    def max_true(
        self,
        low: int,
        high: int,
        predicate: Callable[[int], bool],
    ) -> int:
        if high < low:
            return low - 1

        if not predicate(low):
            return low - 1

        lo = low
        hi = high
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if predicate(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo


class GridChunkSearch(ChunkSearch):
    def __init__(self, *, step: int = 32) -> None:
        if step <= 0:
            raise ValueError("step must be > 0.")
        self.step = int(step)

    def max_true(
        self,
        low: int,
        high: int,
        predicate: Callable[[int], bool],
    ) -> int:
        if high < low:
            return low - 1

        candidates = {low, high}
        step = self.step
        first = ((low + step - 1) // step) * step
        for x in range(first, high + 1, step):
            candidates.add(x)

        best = low - 1
        for x in sorted(candidates):
            if x < low or x > high:
                continue
            if predicate(x):
                best = x
        return best


class ChunkSearchFactory:
    _SEARCH_REGISTRY = {
        "binary": BinaryChunkSearch,
        "grid": GridChunkSearch,
    }

    @classmethod
    def get_search(cls, search_name: str, **kwargs) -> ChunkSearch:
        return cls._SEARCH_REGISTRY[search_name](**kwargs)

    @classmethod
    def get_available_searches(cls) -> List[str]:
        return list(cls._SEARCH_REGISTRY.keys())
