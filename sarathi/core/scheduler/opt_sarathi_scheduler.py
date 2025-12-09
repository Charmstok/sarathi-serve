import time
from typing import List

import numpy as np

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger

logger = init_logger(__name__)


class OptSarathiScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self.enable_dynamic_chunking_schedule = (
            self.scheduler_config.enable_dynamic_chunking_schedule
        )
        # 以下参数仅在动态分块模式下生效
        self.low_chunk_size = self.scheduler_config.low_chunk_size
        self.high_chunk_size = self.scheduler_config.high_chunk_size
        self.chunk_schedule_max_tokens = self.scheduler_config.chunk_schedule_max_tokens
        self.chunk_schedule_stages = self.scheduler_config.chunk_schedule_stages

        if self.enable_dynamic_chunking_schedule:
            assert self.chunk_schedule_stages > 0
            assert self.chunk_schedule_max_tokens > 0
            assert self.low_chunk_size % 32 == 0
            assert self.high_chunk_size % 32 == 0
            self._chunk_sizes = self._compute_chunk_size_schedule()
            self._tokens_per_stage = int(
                np.ceil(self.chunk_schedule_max_tokens / self.chunk_schedule_stages)
            )

    def _compute_chunk_size_schedule(self):
        # 使用 numpy 生成等差数列（ between low_chunk_size and high_chunk_size）
        chunk_sizes = np.linspace(
            self.low_chunk_size,
            self.high_chunk_size,
            self.chunk_schedule_stages,
            dtype=np.int32,
        )[::-1]
        # align each chunk size to the nearest multiple of 32 or self.low_chunk_size
        round_of_chunk_sizes = min(32, self.low_chunk_size)
        chunk_sizes = (
            np.round(chunk_sizes / round_of_chunk_sizes) * round_of_chunk_sizes
        )
        chunk_sizes = chunk_sizes.astype(np.int64).tolist()

        return chunk_sizes

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    # 计算当前请求在本轮迭代中应该运行多少个 token
    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        # 动态 chunk_size 模式
        if self.enable_dynamic_chunking_schedule:
            request_stage_idx = int(
                np.ceil(
                    seq.get_num_prompt_tokens_stage_processed()
                    // self._tokens_per_stage
                )
            )
            assert request_stage_idx < len(self._chunk_sizes)
            chunk_size = self._chunk_sizes[request_stage_idx]
        # 静态 chunk_size 模式
        else:
            chunk_size = self.chunk_size

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed(), # 剩余 prompt 长度
            chunk_size - num_batched_tokens,                                    # 剩余 budget（预算）
        )

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # 记录当前时间，用于优先级排序
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0

        ######################################################################
        # 阶段 1：将已有的运行中序列组加入 batch。(处理正在运行的请求)
        # 存在两种情况：
        # 1. 序列组的 prefill 尚未完成。这类序列的处理流程与 Sarathi 调度器中的逻辑完全一致。
        # 2. 序列组的 prefill 已经完成。在这种情况下，我们需要检查下一段解码 token 的内存可用性，
        #    并在必要时抢占（preempt）一些序列组。
        #
        # 需要注意，被抢占的序列组可能属于上述两类中的任意一种。
        ######################################################################

        # 注意：只有在没有可用槽位，让所有序列组保持 RUNNING 状态时，才会发生抢占。
        # 此时，策略负责决定抢占哪些序列组。
        self.running = self.policy.sort_by_priority(now, self.running)

        # 优先级1：处理 Decode 任务
        # 首轮处理先所有已完成预填充的请求, 以便准确统计解码 token 的数量
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            # 如果请求被暂停，暂不处理
            if not seq.is_paused():
                running.append(seq)
                continue

            # 如果是 Prefill 还没跑完的任务，先存起来，稍后处理
            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                if self.running:
                    # 如果显存不足，执行 _preempt 将低优先级请求踢回等待队列，腾出空间
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    # 如果 self.running 已经空了，说明：
                    # 1. 也就是当前 Batch 里除了我（seq），其他人都被踢光了。
                    # 2. 即使这样，剩下的空间还是不够我放下一个 Token。
                    # 那么，没办法，只能把自己也停掉。
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                # 在物理显存中真正分配这个 slot
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1 # 扣除一个token的预算
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )

        # 优先级2：处理已加入running列表，但只有部分chunk执行了prefill，没有完全执行完Prefill阶段的请求。
        # 现在加入尚未完成预填充的请求，这些预填充所需的内存早已分配完毕，
        # 因此能够一次性全部运行它们
        for seq in running_prefills:
            # 断言：确保这里面装的确实是还没跑完 Prefill 的任务
            assert not seq.prompt_stage_processing_finished

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # 只要该请求先前能放入整个批次，现在也应该能放得下。
            # 因此在非流水线场景下，这个条件永远为假；
            # 但在流水线场景里，不同微批次之间请求的分组可能变化，所以并不保证永远成立。
            if next_num_prefill_tokens == 0:
                running.append(seq)
                continue

            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)

        # 优先级3：处理等待队列的新情求
        ######################################################################
        # Phase 2: 处理等待队列中的新请求 (New Requests)
        # 目标：Piggybacking (捎带执行)，实现 Stall-free Batching
        ######################################################################
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        # 只要还有 Budget (num_batched_tokens < chunk_size)，就尝试加入新请求
        while self.waiting:
            seq = self.waiting[0]

            # 处理 Benchmarking 的特殊情况（模拟请求到达时间）
            if seq.arrival_time > now:
                break

            # 检查 Prompt 是否超过模型最大支持长度
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            # 如果显存不够，这里不会抢占，而是直接不调度该新请求
            if not self.block_manager.can_allocate(seq):
                # this is different from vllm scheduler
                # even if we cannot allocate this sequence group
                # there might be other sequence groups that can be allocated
                break

            # 检查最大并发序列数限制
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # 计算这个新请求能分到多少 Budget
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            # 如果计算结果为 0，说明 num_batched_tokens 已经达到了 chunk_size
            # 此时停止接纳新请求
            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)                       # 真正从等待队列中移除
            self._allocate(seq)                             # 在 Block Manager 中正式分配显存呢
            num_batched_tokens += next_num_prefill_tokens   # 扣除预算
            scheduled_seq_metadata_list.append(             # 告诉引擎，这个新请求本轮只跑 next_num_prefill_tokens 这么长
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)                             # 加入本轮运行名单

        # 将本轮构建好的 running 列表（包含 Phase 1 的未处理完的老请求和 Phase 2 的新请求）赋值给 self.running，作为系统的最新状态。
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
