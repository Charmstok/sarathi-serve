import time
import os
from typing import List

import numpy as np

from sarathi.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    OptSarathiSchedulerConfig,
)
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.chunk_search import ChunkSearchFactory
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.logger import init_logger
from sarathi.time_balance.config import MODEL_CACHE_PATH
from sarathi.time_balance.predict_time import TimePredictor

logger = init_logger(__name__)


class OptSarathiScheduler(BaseScheduler):

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: OptSarathiSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        super().__init__(model_config, scheduler_config, cache_config, parallel_config)

        self.chunk_size = self.scheduler_config.chunk_size
        self.target_time = self.scheduler_config.target_time
        self.chunk_search_granularity = self.scheduler_config.chunk_search_granularity
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

        assert os.path.exists(
            MODEL_CACHE_PATH
        ), f"TimePredictor cache missing: {MODEL_CACHE_PATH}"
        self._time_predictor = TimePredictor.load(MODEL_CACHE_PATH)
        self._chunk_search = ChunkSearchFactory.get_search(
            "grid", step=self.chunk_search_granularity
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

    def _predict_latency_ms(
        self,
        *,
        decode_tokens: int,
        sum_decode_context_len: int,
        batch_request_count: int,
        prefill_tokens: int,
        prefill_processed_tokens: int,
        gpu_mem_used_mb: float,
        gpu_mem_free_mb: float,
        cuda_allocated_mb: float,
        cuda_reserved_mb: float,
    ) -> float:
        return float(
            self._time_predictor.predict(
                decode_tokens=decode_tokens,
                sum_decode_context_len=sum_decode_context_len,
                batch_request_count=batch_request_count,
                prefill_tokens=prefill_tokens,
                prefill_processed_tokens=prefill_processed_tokens,
                gpu_mem_used_mb=gpu_mem_used_mb,
                gpu_mem_free_mb=gpu_mem_free_mb,
                cuda_allocated_mb=cuda_allocated_mb,
                cuda_reserved_mb=cuda_reserved_mb,
            ).item()
        )

    def _get_gpu_mem_features(self) -> tuple[float, float, float, float]:
        stats = self.get_last_runtime_stats()
        if stats is None:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(stats.gpu_mem_used_mb),
            float(stats.gpu_mem_free_mb),
            float(stats.cuda_allocated_mb),
            float(stats.cuda_reserved_mb),
        )

    def _get_seq_prefill_search_high(self, seq: Sequence, num_batched_tokens: int) -> int:
        assert not seq.is_finished()

        if self.enable_dynamic_chunking_schedule:
            request_stage_idx = int(
                np.ceil(
                    seq.get_num_prompt_tokens_stage_processed()
                    // self._tokens_per_stage
                )
            )
            assert request_stage_idx < len(self._chunk_sizes)
            chunk_size = self._chunk_sizes[request_stage_idx]
        else:
            chunk_size = self.chunk_size

        remaining_budget = chunk_size - num_batched_tokens

        remaining_prompt = (
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed()
        )
        if remaining_prompt <= 0 or remaining_budget <= 0:
            return 0

        if remaining_budget < remaining_prompt:
            return remaining_budget
        return remaining_prompt

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

        remaining_budget = chunk_size - num_batched_tokens

        remaining_prompt = (
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed()
        )  # 剩余 prompt 长度
        if remaining_budget < remaining_prompt:
            next_num_tokens = remaining_budget  # 剩余 budget（预算）
        else:
            next_num_tokens = remaining_prompt

        return next_num_tokens

    def _schedule(self) -> SchedulerOutputs:
        # 记录当前时间，用于优先级排序
        now = time.monotonic()

        running: List[Sequence] = []
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        num_batched_tokens: int = 0
        target_time_ms = float(self.scheduler_config.target_time)

        (
            gpu_mem_used_mb,
            gpu_mem_free_mb,
            cuda_allocated_mb,
            cuda_reserved_mb,
        ) = self._get_gpu_mem_features()

        # --- Phase 0: 初始化当前 Batch 的基础负载 (用于增量预测) ---
        current_decode_tokens = 0
        current_sum_decode_context_len = 0
        current_batch_request_count = 0
        current_prefill_tokens = 0
        current_prefill_processed_tokens = 0

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
                current_decode_tokens += 1
                current_sum_decode_context_len += seq.get_len()
                current_batch_request_count += 1

        # 优先级2：处理已加入running列表，但只有部分chunk执行了prefill，没有完全执行完Prefill阶段的请求。
        # 现在加入尚未完成预填充的请求，这些预填充所需的内存早已分配完毕，
        # 因此能够一次性全部运行它们
        for seq in running_prefills:
            # 断言：确保这里面装的确实是还没跑完 Prefill 的任务
            assert not seq.prompt_stage_processing_finished

            high = self._get_seq_prefill_search_high(seq, num_batched_tokens)
            if high == 0:
                running.append(seq)
                continue

            seq_processed = seq.get_num_prompt_tokens_stage_processed()

            def feasible(chunk_size: int) -> bool:
                predicted_ms = self._predict_latency_ms(
                    decode_tokens=current_decode_tokens,
                    sum_decode_context_len=current_sum_decode_context_len,
                    batch_request_count=current_batch_request_count + 1,
                    prefill_tokens=current_prefill_tokens + chunk_size,
                    prefill_processed_tokens=(
                        current_prefill_processed_tokens + seq_processed
                    ),
                    gpu_mem_used_mb=gpu_mem_used_mb,
                    gpu_mem_free_mb=gpu_mem_free_mb,
                    cuda_allocated_mb=cuda_allocated_mb,
                    cuda_reserved_mb=cuda_reserved_mb,
                )
                return predicted_ms <= target_time_ms

            next_num_prefill_tokens = self._chunk_search.max_true(1, high, feasible)

            # 如果本轮剩余 budget 不足（_get_seq_next_num_prefill_tokens 返回 0），
            # 先把该请求放回 running，保持占位，等下一轮再继续 prefill。
            # 在流水线场景或开启最小分块阈值时可能出现这种情况。
            if next_num_prefill_tokens == 0:
                # 避免 Prefill-only 场景下因为 target_time 过低导致永远调度不到 token（死锁）。
                # 若当前 batch 还没有任何任务被调度，则强制至少推进 1 个 token。
                if not scheduled_seq_metadata_list and high >= 1:
                    next_num_prefill_tokens = 1
                else:
                    running.append(seq)
                    continue

            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
            current_prefill_tokens += next_num_prefill_tokens
            current_prefill_processed_tokens += seq_processed
            current_batch_request_count += 1

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
            high = self._get_seq_prefill_search_high(seq, num_batched_tokens)
            if high == 0:
                break

            seq_processed = seq.get_num_prompt_tokens_stage_processed()

            def feasible(chunk_size: int) -> bool:
                predicted_ms = self._predict_latency_ms(
                    decode_tokens=current_decode_tokens,
                    sum_decode_context_len=current_sum_decode_context_len,
                    batch_request_count=current_batch_request_count + 1,
                    prefill_tokens=current_prefill_tokens + chunk_size,
                    prefill_processed_tokens=(
                        current_prefill_processed_tokens + seq_processed
                    ),
                    gpu_mem_used_mb=gpu_mem_used_mb,
                    gpu_mem_free_mb=gpu_mem_free_mb,
                    cuda_allocated_mb=cuda_allocated_mb,
                    cuda_reserved_mb=cuda_reserved_mb,
                )
                return predicted_ms <= target_time_ms

            next_num_prefill_tokens = self._chunk_search.max_true(1, high, feasible)

            # 如果计算结果为 0，说明 num_batched_tokens 已经达到了 chunk_size
            # 此时停止接纳新请求
            if next_num_prefill_tokens == 0:
                # 避免 Prefill-only 场景下因为 target_time 过低导致永远调度不到 token（死锁）。
                # 若当前 batch 还没有任何任务被调度，则强制至少推进 1 个 token。
                if not scheduled_seq_metadata_list and high >= 1:
                    next_num_prefill_tokens = 1
                else:
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
            current_prefill_tokens += next_num_prefill_tokens
            current_prefill_processed_tokens += seq_processed
            current_batch_request_count += 1

        # 将本轮构建好的 running 列表（包含 Phase 1 的未处理完的老请求和 Phase 2 的新请求）赋值给 self.running，作为系统的最新状态。
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
