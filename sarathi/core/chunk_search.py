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
        """Return the maximum x in [low, high] such that predicate(x) is True.

        Assumes predicate is monotonic: True...True, False...False.
        Returns (low - 1) if predicate(low) is False.
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


class ChunkSearchFactory:
    _SEARCH_REGISTRY = {
        "binary": BinaryChunkSearch,
    }

    @classmethod
    def get_search(cls, search_name: str, **kwargs) -> ChunkSearch:
        return cls._SEARCH_REGISTRY[search_name](**kwargs)

    @classmethod
    def get_available_searches(cls) -> List[str]:
        return list(cls._SEARCH_REGISTRY.keys())
