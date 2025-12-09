from typing import List
from abc import ABC, abstractmethod

from sarathi.core.datatypes.sequence import Sequence


class Policy(ABC):

    @abstractmethod
    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        pass

    def sort_by_priority(
        self,
        now: float,
        seqs: List[Sequence],
    ) -> List[Sequence]:
        return sorted(
            seqs,
            key=lambda seq: self.get_priority(now, seq),
            reverse=True,
        )

# 先来先服务
class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        return now - seq.arrival_time

# 最短prompt优先
class SPF(Policy):

    def get_priority(
        self,
        now: float,
        seq: Sequence,
    ) -> float:
        return 1.0 / seq.get_prompt_len()


class PolicyFactory:

    _POLICY_REGISTRY = {
        "fcfs": FCFS,
        "spf": SPF,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)

    @classmethod
    def get_available_policies(cls) -> List[str]:
        return list(cls._POLICY_REGISTRY.keys())
