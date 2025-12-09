# coding=utf-8
# Copyright (c) Alibaba Cloud.
# Copyright (c) Microsoft Corporation.
# Adapted from Sarathi-Serve Qwen implementation for Qwen3/Qwen2 architecture.

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)


class Qwen3MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str = "silu",
    ):
        super().__init__()
        # Qwen3 uses SwiGLU: gate_proj and up_proj are merged for efficiency
        # Merged into: gate_up_proj [2 * intermediate_size]
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(OperationMetrics.MLP_ACTIVATION)

    def forward(self, x):
        # x: [batch, seq, hidden]
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3Attention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position_embeddings: int,
            rope_theta: float = 1000000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            rms_norm_eps: float = 1e-6,
            attention_bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        # Verify TP compatibility
        if self.total_num_heads % tensor_model_parallel_world_size != 0:
            raise ValueError(
                f"Number of heads ({self.total_num_heads}) must be divisible by TP world size ({tensor_model_parallel_world_size})")
        if self.total_num_kv_heads % tensor_model_parallel_world_size != 0:
            raise ValueError(
                f"Number of KV heads ({self.total_num_kv_heads}) must be divisible by TP world size ({tensor_model_parallel_world_size})")

        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.num_kv_heads = self.total_num_kv_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads

        self.scaling = self.head_dim ** -0.5

        # Determine QKV Output Dimension
        # Q: num_heads * head_dim
        # K: num_kv_heads * head_dim
        # V: num_kv_heads * head_dim
        self.q_num_d = self.total_num_heads * self.head_dim
        self.kv_num_d = self.total_num_kv_heads * self.head_dim
        self.qkv_dim = self.q_num_d + 2 * self.kv_num_d

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            self.qkv_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
        )

        # Qwen3 Specific: Q-Norm and K-Norm applied BEFORE RoPE
        # Use standard RMSNorm but apply it per-head manually in forward
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            layer_cache_idx: int,
            attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        # 1. QKV Projection
        # qkv shape: [num_tokens, local_qkv_dim]
        qkv, _ = self.qkv_proj(hidden_states)

        # 2. Split Q, K, V based on local head counts
        local_q_dim = self.num_heads * self.head_dim
        local_kv_dim = self.num_kv_heads * self.head_dim

        q, k, v = torch.split(qkv, [local_q_dim, local_kv_dim, local_kv_dim], dim=-1)

        # 3. Apply Q-Norm and K-Norm
        # Qwen3 applies norm on the head_dim.
        # We need to reshape: [tokens, heads*head_dim] -> [tokens*heads, head_dim]
        q_shape_orig = q.shape
        k_shape_orig = k.shape

        # Reshape to flatten heads -> [tokens * num_heads, head_dim]
        q = q.contiguous().view(-1, self.head_dim)
        k = k.contiguous().view(-1, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape back to [tokens, total_dim]
        q = q.view(q_shape_orig)
        k = k.view(k_shape_orig)

        # 4. RoPE
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        # 5. Backend Attention (FlashInfer/XFormers)
        # This handles PagedAttention and Chunked Prefill logic
        attn_output = attention_backend_wrapper.forward(
            q,
            k,
            v,
            layer_cache_idx,
            self.scaling,
        )

        # 6. Output Projection
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_scaling = getattr(config, "rope_scaling", None)
        attention_bias = getattr(config, "attention_bias", False)

        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=attention_bias,
        )

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            layer_cache_idx: int,
            attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        # Self Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            layer_cache_idx=layer_cache_idx,
            attention_backend_wrapper=attention_backend_wrapper,
        )
        hidden_states = residual + hidden_states

        # MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_tokens = None
        if is_pipeline_first_stage():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, perform_initialization=False
            )

        num_layers = config.num_hidden_layers
        pp_world_size = get_pipeline_model_parallel_world_size()
        assert num_layers % pp_world_size == 0
        num_layers_per_stage = num_layers // pp_world_size

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, i)
                for i in range(num_layers_per_stage)
            ]
        )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions, hidden_states, i, attention_backend_wrapper
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
            self,
            hidden_states: torch.Tensor,
            positions: torch.Tensor,
            attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], self.config.hidden_size),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(
            hidden_states, positions, attention_backend_wrapper
        )

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    # Weights for Tensor Parallelism
    _column_parallel_weights = [
        "qkv_proj.weight", "qkv_proj.bias",
        "gate_up_proj.weight",
        "lm_head.weight",
        "embed_tokens.weight"
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def load_weights(
            self,
            model_name_or_path: str,
            cache_dir: Optional[str] = None,
            load_format: str = "auto",
            revision: Optional[str] = None,
    ):
        tp_world_size = get_tensor_model_parallel_world_size()
        pp_world_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        assert self.config.num_hidden_layers % pp_world_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_world_size
        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        state_dict = self.state_dict()

        # Buffers for weight merging (Dynamic buffering)
        loaded_gate_up_buffer = {}
        loaded_qkv_buffer = {}

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            # Map global layers to local stage layers
            if "model.layers." in name:
                try:
                    layer_id = int(name.split(".")[2])
                except ValueError:
                    continue

                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue
                new_layer_id = layer_id - first_layer_id
                name = name.replace(f"layers.{layer_id}", f"layers.{new_layer_id}")

            # PP: Skip weights not on this stage
            if pp_rank != 0 and "embed_tokens" in name:
                continue
            if pp_rank != pp_world_size - 1 and ("lm_head" in name or "norm" in name):
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            # --- Logic 1: Merge MLP (gate + up -> gate_up) ---
            if "gate_proj" in name or "up_proj" in name:
                layer_key = name.split("mlp")[0]
                if layer_key not in loaded_gate_up_buffer:
                    loaded_gate_up_buffer[layer_key] = {}

                type_key = "gate" if "gate_proj" in name else "up"
                loaded_gate_up_buffer[layer_key][type_key] = loaded_weight

                # If both parts are ready, merge and load
                if "gate" in loaded_gate_up_buffer[layer_key] and "up" in loaded_gate_up_buffer[layer_key]:
                    gate = loaded_gate_up_buffer[layer_key]["gate"]
                    up = loaded_gate_up_buffer[layer_key]["up"]
                    merged_weight = torch.cat([gate, up], dim=0)

                    target_name = layer_key + "mlp.gate_up_proj.weight"
                    if target_name in state_dict:
                        param = state_dict[target_name]
                        load_tensor_parallel_weights(
                            param, merged_weight, target_name, self._column_parallel_weights,
                            self._row_parallel_weights, tp_rank
                        )
                    del loaded_gate_up_buffer[layer_key]
                continue

            # --- Logic 2: Merge Attention (q + k + v -> qkv) ---
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                layer_key = name.split("self_attn")[0]
                if layer_key not in loaded_qkv_buffer:
                    loaded_qkv_buffer[layer_key] = {"weight": {}, "bias": {}}

                is_bias = "bias" in name
                suffix = "bias" if is_bias else "weight"

                if "q_proj" in name:
                    type_key = "q"
                elif "k_proj" in name:
                    type_key = "k"
                elif "v_proj" in name:
                    type_key = "v"

                loaded_qkv_buffer[layer_key][suffix][type_key] = loaded_weight

                # If q, k, v are all ready for this suffix
                buffer = loaded_qkv_buffer[layer_key][suffix]
                if "q" in buffer and "k" in buffer and "v" in buffer:
                    merged = torch.cat([buffer["q"], buffer["k"], buffer["v"]], dim=0)
                    target_name = layer_key + f"self_attn.qkv_proj.{suffix}"

                    if target_name in state_dict:
                        param = state_dict[target_name]
                        load_tensor_parallel_weights(
                            param, merged, target_name, self._column_parallel_weights, self._row_parallel_weights,
                            tp_rank
                        )
                    loaded_qkv_buffer[layer_key][suffix] = {}
                continue

            # --- Logic 3: Direct Mapping ---
            if "embed_tokens" in name:
                target_name = "model.embed_tokens.weight"
            elif "lm_head" in name:
                target_name = "lm_head.weight"
            elif "model.norm" in name:
                target_name = "model.norm.weight"
            elif "input_layernorm" in name:
                target_name = name
            elif "post_attention_layernorm" in name:
                target_name = name
            elif "down_proj" in name:
                target_name = name.replace("mlp.down_proj", "mlp.down_proj")
            elif "o_proj" in name:
                target_name = name.replace("self_attn.o_proj", "self_attn.o_proj")
            elif "q_norm" in name:
                # Qwen3 Specific: Load q_norm weight
                target_name = name.replace("self_attn.q_norm", "self_attn.q_norm")
            elif "k_norm" in name:
                # Qwen3 Specific: Load k_norm weight
                target_name = name.replace("self_attn.k_norm", "self_attn.k_norm")
            else:
                target_name = name

            if target_name in state_dict:
                param = state_dict[target_name]
                if "embed_tokens" in target_name or "lm_head" in target_name:
                    load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                else:
                    load_tensor_parallel_weights(
                        param,
                        loaded_weight,
                        target_name,
                        self._column_parallel_weights,
                        self._row_parallel_weights,
                        tp_rank,
                    )