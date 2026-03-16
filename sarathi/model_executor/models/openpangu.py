# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openPangu 模型
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

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


def _get_num_kv_heads(config: PretrainedConfig) -> int:
    return getattr(config, "num_key_value_heads", config.num_attention_heads)


def _get_dtype(config: PretrainedConfig) -> torch.dtype:
    return getattr(config, "dtype", getattr(config, "torch_dtype", torch.get_default_dtype()))


def _check_hidden_act(hidden_act: str) -> None:
    if hidden_act != "silu":
        raise ValueError(
            f"Unsupported activation: {hidden_act}. Only silu is supported for now."
        )


class OpenPanguMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        bias: bool = False,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        _check_hidden_act(hidden_act)
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(
            OperationMetrics.MLP_ACTIVATION,
            layer_id=layer_id,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
class OpenPanguAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        o_proj_bias: bool = False,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads

        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Number of heads ({self.total_num_heads}) must be divisible by TP world size ({tp_size})."
            )
        if self.total_num_kv_heads % tp_size != 0:
            raise ValueError(
                f"Number of KV heads ({self.total_num_kv_heads}) must be divisible by TP world size ({tp_size})."
            )

        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_id = layer_id

        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=attention_bias,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=o_proj_bias,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE, layer_id=layer_id)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = attention_backend_wrapper.forward(
            q,
            k,
            v,
            layer_cache_idx,
            self.scaling,
            self.layer_id,
        )
        output, _ = self.o_proj(attn_output)
        return output


class OpenPanguDecoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig, layer_id: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        attention_bias = getattr(config, "qkv_bias", None)
        if attention_bias is None:
            attention_bias = getattr(config, "attention_bias", getattr(config, "bias", False))
        o_proj_bias = getattr(config, "attention_bias", getattr(config, "bias", False))

        self.self_attn = OpenPanguAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=_get_num_kv_heads(config),
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            o_proj_bias=o_proj_bias,
            layer_id=layer_id,
        )

        self.mlp = OpenPanguMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            bias=getattr(config, "mlp_bias", False),
            layer_id=layer_id,
        )

        eps = getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=eps,
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=eps,
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            layer_cache_idx=layer_cache_idx,
            attention_backend_wrapper=attention_backend_wrapper,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OpenPanguModel(nn.Module):

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((config.vocab_size + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                config.hidden_size,
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        num_layers = config.num_hidden_layers // get_pipeline_model_parallel_world_size()
        layer_offset = get_pipeline_model_parallel_rank() * num_layers
        self.layers = nn.ModuleList(
            [
                OpenPanguDecoderLayer(config, layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.norm = None
        if is_pipeline_last_stage():
            eps = getattr(config, "rms_norm_eps", getattr(config, "layer_norm_epsilon", 1e-6))
            self.norm = RMSNorm(config.hidden_size, eps=eps)

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
                positions,
                hidden_states,
                i,
                attention_backend_wrapper,
            )

        if self.norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class OpenPanguForCausalLM(nn.Module):
    _column_parallel_weights = [
        "qkv_proj.weight",
        "qkv_proj.bias",
        "gate_up_proj.weight",
        "gate_up_proj.bias",
    ]
    _row_parallel_weights = ["o_proj.weight", "down_proj.weight"]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.model = OpenPanguModel(config)
        vocab_size = ((config.vocab_size + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
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
                dtype=_get_dtype(self.config),
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, attention_backend_wrapper)

        if not self.is_pipeline_last_stage:
            send(hidden_states)
        return hidden_states

    def _load_merged_projection(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        loaded_buffers: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
        state_dict: Dict[str, torch.Tensor],
        tp_rank: int,
    ) -> bool:
        if any(part in name for part in ["q_proj", "k_proj", "v_proj"]):
            layer_key = name.split("self_attn")[0]
            buffer_entry = loaded_buffers.setdefault(layer_key, {"weight": {}, "bias": {}})
            suffix = "bias" if name.endswith(".bias") else "weight"
            if "q_proj" in name:
                part = "q"
            elif "k_proj" in name:
                part = "k"
            else:
                part = "v"
            buffer_entry[suffix][part] = loaded_weight
            if len(buffer_entry[suffix]) == 3:
                merged = torch.cat(
                    [
                        buffer_entry[suffix]["q"],
                        buffer_entry[suffix]["k"],
                        buffer_entry[suffix]["v"],
                    ],
                    dim=0,
                )
                target_name = layer_key + f"self_attn.qkv_proj.{suffix}"
                if target_name in state_dict:
                    load_tensor_parallel_weights(
                        state_dict[target_name],
                        merged,
                        target_name,
                        self._column_parallel_weights,
                        self._row_parallel_weights,
                        tp_rank,
                    )
                buffer_entry[suffix] = {}
            return True

        if any(part in name for part in ["gate_proj", "up_proj"]):
            layer_key = name.rsplit(".", 2)[0]
            buffer_entry = loaded_buffers.setdefault(layer_key, {"weight": {}, "bias": {}})
            suffix = "bias" if name.endswith(".bias") else "weight"
            part = "gate" if "gate_proj" in name else "up"
            buffer_entry[suffix][part] = loaded_weight
            if len(buffer_entry[suffix]) == 2:
                merged = torch.cat(
                    [buffer_entry[suffix]["gate"], buffer_entry[suffix]["up"]],
                    dim=0,
                )
                target_name = layer_key + f".gate_up_proj.{suffix}"
                if target_name in state_dict:
                    load_tensor_parallel_weights(
                        state_dict[target_name],
                        merged,
                        target_name,
                        self._column_parallel_weights,
                        self._row_parallel_weights,
                        tp_rank,
                    )
                buffer_entry[suffix] = {}
            return True

        return False

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ) -> None:
        pp_world_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        assert self.config.num_hidden_layers % pp_world_size == 0
        layers_per_stage = self.config.num_hidden_layers // pp_world_size
        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        state_dict = self.state_dict()
        loaded_qkv_buffer: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        loaded_gate_up_buffer: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path,
            cache_dir,
            load_format,
            revision,
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            if getattr(self.config, "tie_word_embeddings", False) and name == "lm_head.weight":
                continue

            if "model.layers." in name:
                try:
                    layer_id = int(name.split(".")[2])
                except ValueError:
                    continue
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue
                new_layer_id = layer_id - first_layer_id
                name = name.replace(f"layers.{layer_id}", f"layers.{new_layer_id}")

            if pp_rank != 0 and "embed_tokens" in name:
                continue
            if pp_rank != pp_world_size - 1 and (
                "lm_head" in name or name == "model.norm.weight"
            ):
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            if self._load_merged_projection(
                name,
                loaded_weight,
                loaded_qkv_buffer,
                state_dict,
                tp_rank,
            ):
                continue
            if self._load_merged_projection(
                name,
                loaded_weight,
                loaded_gate_up_buffer,
                state_dict,
                tp_rank,
            ):
                continue

            if name == "embed_tokens.weight":
                target_name = "model.embed_tokens.weight"
            elif name == "norm.weight":
                target_name = "model.norm.weight"
            else:
                target_name = name

            if target_name not in state_dict:
                continue

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


class PanguEmbeddedForCausalLM(OpenPanguForCausalLM):
    pass
