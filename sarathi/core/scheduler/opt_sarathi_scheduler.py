import time
import os
from typing import List, Optional

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
from sarathi.core.datatypes.sequence_status import SequenceStatus
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
        self.chunk_score_underfill_penalty = (
            self.scheduler_config.chunk_score_underfill_penalty
        )
        self.chunk_score_overflow_penalty = (
            self.scheduler_config.chunk_score_overflow_penalty
        )
        self.enable_prefill_slot_reservation = (
            self.scheduler_config.enable_prefill_slot_reservation
        )
        self.prefill_reserve_waiting_threshold = (
            self.scheduler_config.prefill_reserve_waiting_threshold
        )
        self.prefill_reserved_seq_slots = (
            self.scheduler_config.prefill_reserved_seq_slots
        )
        self.enable_active_prefill_control = (
            self.scheduler_config.enable_active_prefill_control
        )
        self.max_active_prefill_seqs = (
            self.scheduler_config.max_active_prefill_seqs
        )
        self.min_active_prefill_chunk_size = (
            self.scheduler_config.min_active_prefill_chunk_size
        )
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
        max_decode_context_len: int,
        max_prefill_processed_tokens: int,
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
                max_decode_context_len=max_decode_context_len,
                max_prefill_processed_tokens=max_prefill_processed_tokens,
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

    def _chunk_target_score(
        self,
        *,
        predicted_ms: float,
        target_time_ms: float,
    ) -> float:
        if predicted_ms <= target_time_ms:
            return (
                (target_time_ms - predicted_ms) * self.chunk_score_underfill_penalty
            )
        return (
            (predicted_ms - target_time_ms) * self.chunk_score_overflow_penalty
        )

    def _get_num_ready_waiting_seqs(self, now: float) -> int:
        ready_waiting = 0
        for seq in self.waiting:
            if seq.arrival_time > now:
                break
            ready_waiting += 1
        return ready_waiting

    def _get_prefill_slot_borrow_limit(self, now: float) -> int:
        if not self.enable_prefill_slot_reservation:
            return 0

        ready_waiting = self._get_num_ready_waiting_seqs(now)
        if ready_waiting < self.prefill_reserve_waiting_threshold:
            return 0

        return min(self.prefill_reserved_seq_slots, ready_waiting)

    def _get_batch_token_cap(self) -> int:
        return int(
            self.scheduler_config.get_max_num_batched_tokens(
                self.model_config.max_model_len
            )
        )

    def _get_min_meaningful_prefill_tokens(self, seq: Sequence) -> int:
        remaining_prompt = (
            seq.get_prompt_len() - seq.get_num_prompt_tokens_stage_processed()
        )
        if remaining_prompt <= 0:
            return 0
        if not self.enable_active_prefill_control:
            return 1
        return min(remaining_prompt, self.min_active_prefill_chunk_size)

    def _get_active_prefill_seq_cap(
        self,
        *,
        decode_seq_count: int,
        num_batched_tokens: int,
    ) -> int:
        remaining_seq_slots = max(
            self.scheduler_config.max_num_seqs - decode_seq_count,
            0,
        )
        if not self.enable_active_prefill_control:
            return remaining_seq_slots
        if remaining_seq_slots == 0:
            return 0
        remaining_token_budget = max(
            self._get_batch_token_cap() - num_batched_tokens,
            0,
        )
        token_limited_cap = remaining_token_budget // self.min_active_prefill_chunk_size
        return min(
            self.max_active_prefill_seqs,
            remaining_seq_slots,
            token_limited_cap,
        )

    def _defer_running_prefill(
        self,
        seq: Sequence,
        deferred_prefills: List[Sequence],
    ) -> None:
        self._free_seq(seq)
        if not seq.is_waiting():
            seq.set_status(SequenceStatus.WAITING)
        deferred_prefills.append(seq)

    def _select_prefill_slot_borrow_victim(
        self,
        running: List[Sequence],
    ) -> Optional[Sequence]:
        for seq in reversed(running):
            if seq.is_paused() and seq.prompt_stage_processing_finished:
                return seq
        return None

    def _remove_scheduled_seq_metadata(
        self,
        seq_id: str,
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
    ) -> None:
        for idx in range(len(scheduled_seq_metadata_list) - 1, -1, -1):
            if scheduled_seq_metadata_list[idx].seq_id == seq_id:
                scheduled_seq_metadata_list.pop(idx)
                return
        raise AssertionError(f"Scheduled metadata missing for seq_id={seq_id}")

    def _borrow_prefill_seq_slot(
        self,
        *,
        victim_seq: Sequence,
        running: List[Sequence],
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata],
        scheduled_decode_seqs: List[Sequence],
        preempted_seq_ids: List[str],
    ) -> None:
        self._free_seq(victim_seq)
        victim_seq.reset_for_recompute()
        preempted_seq_ids.append(victim_seq.seq_id)
        running.remove(victim_seq)
        scheduled_decode_seqs.remove(victim_seq)
        self._remove_scheduled_seq_metadata(
            victim_seq.seq_id, scheduled_seq_metadata_list
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
        scheduled_decode_seqs: List[Sequence] = []
        borrowed_preempted: List[Sequence] = []
        deferred_prefills: List[Sequence] = []

        num_batched_tokens: int = 0
        target_time_ms = float(self.scheduler_config.target_time)
        borrowed_prefill_seq_slots = 0
        active_prefill_seq_count = 0
        active_prefill_seq_cap = 0
        deferred_prefill_seq_count = 0
        waiting_prefill_blocked_by_cap = 0
        waiting_prefill_blocked_by_min_chunk = 0

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
        current_max_decode_context_len = 0
        current_max_prefill_processed_tokens = 0

        self.running = self.policy.sort_by_priority(now, self.running)

        # 优先级1：处理 Decode 任务
        running_prefills: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_stage_processing_finished:
                running_prefills.append(seq)
                continue

            while not self.block_manager.can_append_slot():
                if self.running:
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                else:
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    break
            else:
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                scheduled_decode_seqs.append(seq)
                current_decode_tokens += 1
                seq_context_len = seq.get_len()
                current_sum_decode_context_len += seq_context_len
                current_batch_request_count += 1
                current_max_decode_context_len = max(
                    current_max_decode_context_len, seq_context_len
                )

        # 优先级2：处理已有 unfinished prefill
        for seq in running_prefills:
            assert not seq.prompt_stage_processing_finished

            current_active_prefill_cap = self._get_active_prefill_seq_cap(
                decode_seq_count=len(scheduled_decode_seqs),
                num_batched_tokens=num_batched_tokens,
            )
            active_prefill_seq_cap = current_active_prefill_cap

            if (
                self.enable_active_prefill_control
                and active_prefill_seq_count >= current_active_prefill_cap
            ):
                self._defer_running_prefill(seq, deferred_prefills)
                deferred_prefill_seq_count += 1
                continue

            high = self._get_seq_prefill_search_high(seq, num_batched_tokens)
            if high == 0:
                if self.enable_active_prefill_control:
                    self._defer_running_prefill(seq, deferred_prefills)
                    deferred_prefill_seq_count += 1
                else:
                    running.append(seq)
                continue

            seq_processed = seq.get_num_prompt_tokens_stage_processed()

            def score(chunk_size: int) -> float:
                predicted_ms = self._predict_latency_ms(
                    decode_tokens=current_decode_tokens,
                    sum_decode_context_len=current_sum_decode_context_len,
                    batch_request_count=current_batch_request_count + 1,
                    prefill_tokens=current_prefill_tokens + chunk_size,
                    prefill_processed_tokens=(
                        current_prefill_processed_tokens + seq_processed
                    ),
                    max_decode_context_len=current_max_decode_context_len,
                    max_prefill_processed_tokens=max(
                        current_max_prefill_processed_tokens, seq_processed
                    ),
                    gpu_mem_used_mb=gpu_mem_used_mb,
                    gpu_mem_free_mb=gpu_mem_free_mb,
                    cuda_allocated_mb=cuda_allocated_mb,
                    cuda_reserved_mb=cuda_reserved_mb,
                )
                return self._chunk_target_score(
                    predicted_ms=predicted_ms,
                    target_time_ms=target_time_ms,
                )

            next_num_prefill_tokens = self._chunk_search.min_score(1, high, score)
            min_meaningful_prefill_tokens = self._get_min_meaningful_prefill_tokens(seq)
            forced_prefill_tokens = min(high, min_meaningful_prefill_tokens)
            can_force_progress = high >= 1 and (
                not scheduled_seq_metadata_list
                or (
                    self.enable_active_prefill_control
                    and active_prefill_seq_count == 0
                )
            )

            if next_num_prefill_tokens == 0:
                if self.enable_active_prefill_control:
                    if can_force_progress:
                        next_num_prefill_tokens = forced_prefill_tokens
                    else:
                        self._defer_running_prefill(seq, deferred_prefills)
                        deferred_prefill_seq_count += 1
                        continue
                elif not scheduled_seq_metadata_list and high >= 1:
                    next_num_prefill_tokens = 1
                else:
                    running.append(seq)
                    continue

            if (
                self.enable_active_prefill_control
                and next_num_prefill_tokens < min_meaningful_prefill_tokens
            ):
                if can_force_progress:
                    next_num_prefill_tokens = forced_prefill_tokens
                else:
                    self._defer_running_prefill(seq, deferred_prefills)
                    deferred_prefill_seq_count += 1
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
            current_max_prefill_processed_tokens = max(
                current_max_prefill_processed_tokens, seq_processed
            )
            active_prefill_seq_count += 1

        # 优先级3：处理 waiting 中的新 prefill 请求
        while self.waiting:
            seq = self.waiting[0]

            if seq.arrival_time > now:
                break

            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                continue

            if not self.block_manager.can_allocate(seq):
                break

            current_active_prefill_cap = self._get_active_prefill_seq_cap(
                decode_seq_count=len(scheduled_decode_seqs),
                num_batched_tokens=num_batched_tokens,
            )
            active_prefill_seq_cap = current_active_prefill_cap
            if (
                self.enable_active_prefill_control
                and active_prefill_seq_count >= current_active_prefill_cap
            ):
                waiting_prefill_blocked_by_cap += 1
                break

            if len(running) >= self.scheduler_config.max_num_seqs:
                borrow_limit = self._get_prefill_slot_borrow_limit(now)
                if borrowed_prefill_seq_slots >= borrow_limit:
                    break

                victim_seq = self._select_prefill_slot_borrow_victim(running)
                if victim_seq is None:
                    break

                high_if_borrow = self._get_seq_prefill_search_high(
                    seq, max(num_batched_tokens - 1, 0)
                )
                if high_if_borrow == 0:
                    break

                victim_context_len = victim_seq.get_len()
                self._borrow_prefill_seq_slot(
                    victim_seq=victim_seq,
                    running=running,
                    scheduled_seq_metadata_list=scheduled_seq_metadata_list,
                    scheduled_decode_seqs=scheduled_decode_seqs,
                    preempted_seq_ids=preempted_seq_ids,
                )
                borrowed_preempted.append(victim_seq)
                borrowed_prefill_seq_slots += 1
                num_batched_tokens -= 1
                current_decode_tokens -= 1
                current_sum_decode_context_len -= victim_context_len
                current_batch_request_count -= 1
                current_max_decode_context_len = max(
                    (scheduled_seq.get_len() for scheduled_seq in scheduled_decode_seqs),
                    default=0,
                )

            high = self._get_seq_prefill_search_high(seq, num_batched_tokens)
            if high == 0:
                break

            seq_processed = seq.get_num_prompt_tokens_stage_processed()

            def score(chunk_size: int) -> float:
                predicted_ms = self._predict_latency_ms(
                    decode_tokens=current_decode_tokens,
                    sum_decode_context_len=current_sum_decode_context_len,
                    batch_request_count=current_batch_request_count + 1,
                    prefill_tokens=current_prefill_tokens + chunk_size,
                    prefill_processed_tokens=(
                        current_prefill_processed_tokens + seq_processed
                    ),
                    max_decode_context_len=current_max_decode_context_len,
                    max_prefill_processed_tokens=max(
                        current_max_prefill_processed_tokens, seq_processed
                    ),
                    gpu_mem_used_mb=gpu_mem_used_mb,
                    gpu_mem_free_mb=gpu_mem_free_mb,
                    cuda_allocated_mb=cuda_allocated_mb,
                    cuda_reserved_mb=cuda_reserved_mb,
                )
                return self._chunk_target_score(
                    predicted_ms=predicted_ms,
                    target_time_ms=target_time_ms,
                )

            next_num_prefill_tokens = self._chunk_search.min_score(1, high, score)
            min_meaningful_prefill_tokens = self._get_min_meaningful_prefill_tokens(seq)
            forced_prefill_tokens = min(high, min_meaningful_prefill_tokens)
            can_force_progress = high >= 1 and (
                not scheduled_seq_metadata_list
                or (
                    self.enable_active_prefill_control
                    and active_prefill_seq_count == 0
                )
            )

            if next_num_prefill_tokens == 0:
                if self.enable_active_prefill_control:
                    if can_force_progress:
                        next_num_prefill_tokens = forced_prefill_tokens
                    else:
                        waiting_prefill_blocked_by_min_chunk += 1
                        break
                elif not scheduled_seq_metadata_list and high >= 1:
                    next_num_prefill_tokens = 1
                else:
                    break

            if (
                self.enable_active_prefill_control
                and next_num_prefill_tokens < min_meaningful_prefill_tokens
            ):
                if can_force_progress:
                    next_num_prefill_tokens = forced_prefill_tokens
                else:
                    waiting_prefill_blocked_by_min_chunk += 1
                    break

            seq = self.waiting.pop(0)
            self._allocate(seq)
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
            current_max_prefill_processed_tokens = max(
                current_max_prefill_processed_tokens, seq_processed
            )
            active_prefill_seq_count += 1

        if borrowed_preempted:
            self.waiting.extend(borrowed_preempted)
        if deferred_prefills:
            self.waiting = deferred_prefills + self.waiting

        self.running = running
        active_prefill_seq_cap = self._get_active_prefill_seq_cap(
            decode_seq_count=len(scheduled_decode_seqs),
            num_batched_tokens=num_batched_tokens,
        )

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            active_prefill_seq_cap=active_prefill_seq_cap,
            active_prefill_seq_count=active_prefill_seq_count,
            deferred_prefill_seq_count=deferred_prefill_seq_count,
            waiting_prefill_blocked_by_cap=waiting_prefill_blocked_by_cap,
            waiting_prefill_blocked_by_min_chunk=waiting_prefill_blocked_by_min_chunk,
        )
