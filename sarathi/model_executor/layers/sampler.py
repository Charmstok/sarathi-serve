"""A layer that samples the next tokens from the model's outputs."""

import os
from typing import Dict, List, Optional, Tuple

import flashinfer.sampling
import torch
import torch.nn as nn
from flashinfer.sampling import sampling_from_probs as flashinfer_sampling_from_probs
from flashinfer.sampling import (
    top_k_top_p_sampling_from_logits as flashinfer_top_k_top_p_sampling_from_logits,
)

from sarathi.core.datatypes.sampling_params import SamplingType
from sarathi.core.datatypes.sequence import (
    SamplerOutput,
    SamplerOutputs,
    SequenceMetadata,
)
from sarathi.logger import init_logger
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
)

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TOP_K_ROUND = 32


class Sampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, embedding: torch.Tensor, vocab_size: int) -> None:
        super().__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_metadata_list: List[SequenceMetadata],
    ) -> SamplerOutputs:
        # Get the hidden states that we use for sampling.
        hidden_states = _prune_hidden_states(hidden_states, seq_metadata_list)

        # Get the logits for the next tokens.
        logits = _get_logits(hidden_states, self.embedding, self.vocab_size)

        # Apply temperature scaling.
        temperatures = _get_temperatures(seq_metadata_list)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures, dtype=logits.dtype, device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(seq_metadata_list, self.vocab_size)
        assert len(top_ps) == len(top_ks) == logits.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)

        # Sampling backend:
        # - flashinfer: fast CUDA kernels, but can be less robust on some setups.
        # - torch: slower but more robust, useful for debugging/stability.
        backend = os.environ.get("SARATHI_SAMPLING_BACKEND", "flashinfer").lower()

        if backend == "torch":
            next_token_ids = _sample_with_torch(
                logits=logits,
                seq_metadata_list=seq_metadata_list,
                top_ps=top_ps,
                top_ks=top_ks,
                vocab_size=self.vocab_size,
            ).cpu()
        else:
            if not do_top_p and not do_top_k:
                probs = torch.softmax(logits, dim=-1, dtype=torch.float)
                next_token_ids = _sanitize_sampled_token_ids(
                    _sample_with_flashinfer(probs).view(-1), self.vocab_size
                ).cpu()
            else:
                top_ps_tensor = torch.tensor(
                    top_ps, dtype=logits.dtype, device=logits.device
                )
                top_ks_tensor = torch.tensor(top_ks, dtype=torch.int, device=logits.device)

                next_token_ids = _sanitize_sampled_token_ids(
                    _top_k_top_p_with_flashinfer(logits, top_ks_tensor, top_ps_tensor),
                    self.vocab_size,
                ).cpu()

        return [
            SamplerOutput(seq_metadata_list[i].seq.seq_id, int(next_token_ids[i].item()))
            for i in range(len(seq_metadata_list))
        ]


def _get_logits(
    hidden_states: torch.Tensor, embedding: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    logits = gather_from_tensor_model_parallel_region(logits)
    # Remove paddings in vocab (if any).
    logits = logits[:, :vocab_size]
    return logits


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    seq_metadata_list: List[SequenceMetadata],
) -> torch.Tensor:
    last_token_indices = []
    token_idx = 0
    for seq_metadata in seq_metadata_list:
        if seq_metadata.is_prompt:
            prompt_len = seq_metadata.prompt_chunk_len
            last_token_indices.append(token_idx + prompt_len - 1)
            token_idx += prompt_len
        else:
            last_token_indices.append(token_idx)
            token_idx += 1

    last_token_indices = torch.tensor(
        last_token_indices, dtype=torch.long, device=hidden_states.device
    )
    return hidden_states.index_select(0, last_token_indices)


def _get_temperatures(seq_metadata_list: List[SequenceMetadata]) -> List[float]:
    # Collect the temperatures for the logits.
    temperatures: List[float] = []
    for seq_metadata in seq_metadata_list:
        temperature = seq_metadata.seq.sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            # NOTE: Zero temperature means deterministic sampling
            # (i.e., greedy sampling or beam search).
            # Set the temperature to 1 to avoid division by zero.
            temperature = 1.0
        temperatures.append(temperature)
    return temperatures


def _get_top_p_top_k(
    seq_metadata_list: List[SequenceMetadata],
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for seq_metadata in seq_metadata_list:
        top_p = seq_metadata.seq.sampling_params.top_p
        # k should not be greater than the vocab size.
        top_k = min(seq_metadata.seq.sampling_params.top_k, vocab_size)
        # k=-1 means no truncation.
        top_k = vocab_size if top_k == -1 else top_k
        top_ps.append(top_p)
        top_ks.append(top_k)
    return top_ps, top_ks


def _top_k_top_p_with_flashinfer(
    logits: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
) -> torch.Tensor:
    batch_next_token_ids = flashinfer_top_k_top_p_sampling_from_logits(
        logits, top_ks, top_ps
    )
    return batch_next_token_ids.view(-1)


def _sample_with_flashinfer(probs: torch.Tensor) -> torch.Tensor:
    samples = flashinfer_sampling_from_probs(probs)
    return samples


def _sanitize_sampled_token_ids(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Clamps invalid sampled token IDs to avoid downstream GPU index OOB.

    In rare cases (e.g., NaNs in logits or a sampling kernel glitch), the sampler
    may return token IDs outside [0, vocab_size). Those IDs will later be fed
    back to the model as input tokens and can crash CUDA embedding/gather ops.
    """
    token_ids = token_ids.to(torch.long)
    invalid = (token_ids < 0) | (token_ids >= vocab_size)
    if torch.any(invalid):
        num_invalid = int(invalid.sum().item())
        # Synchronize only in the exceptional path to keep the fast path cheap.
        min_id = int(token_ids.min().item())
        max_id = int(token_ids.max().item())
        logger.warning(
            "Sampler produced out-of-range token ids "
            f"(count={num_invalid}, min={min_id}, max={max_id}, vocab_size={vocab_size}); "
            "replacing invalid ids with 0."
        )
        token_ids = torch.where(invalid, torch.zeros_like(token_ids), token_ids)
    return token_ids


def _sample_with_torch(
    logits: torch.Tensor,
    seq_metadata_list: List[SequenceMetadata],
    top_ps: List[float],
    top_ks: List[int],
    vocab_size: int,
) -> torch.Tensor:
    # Keep sampling well-defined even if logits contain NaN/Inf.
    if not torch.isfinite(logits).all():
        logger.warning(
            "Non-finite logits detected; replacing NaN/Inf with 0 before sampling."
        )
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

    batch_size = logits.shape[0]
    token_ids: List[int] = []

    for i in range(batch_size):
        row_logits = logits[i]

        temperature = seq_metadata_list[i].seq.sampling_params.temperature
        if temperature < _SAMPLING_EPS:
            token_ids.append(int(torch.argmax(row_logits).item()))
            continue

        k = top_ks[i]
        p = top_ps[i]

        # top-k filtering
        if k != vocab_size:
            row_topk_logits, row_topk_indices = torch.topk(row_logits, k)
            filtered_logits = torch.full_like(row_logits, float("-inf"))
            filtered_logits.scatter_(0, row_topk_indices, row_topk_logits)
            row_logits = filtered_logits

        # top-p (nucleus) filtering
        if p < 1.0 - _SAMPLING_EPS:
            sorted_logits, sorted_indices = torch.sort(row_logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > p
            if mask.numel() > 0:
                mask[0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            row_logits = row_logits.scatter(0, sorted_indices, sorted_logits)

        probs = torch.softmax(row_logits, dim=-1, dtype=torch.float)
        sampled = torch.multinomial(probs, num_samples=1)
        token_ids.append(int(sampled.item()))

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    return _sanitize_sampled_token_ids(token_ids_tensor, vocab_size)
