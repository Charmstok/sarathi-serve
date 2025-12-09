# Copyright (c) Alibaba Cloud.
# LICENSE: http://www.apache.org/licenses/LICENSE-2.0

from transformers import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        bos_token_id=151643,
        eos_token_id=151645,
        head_dim=128,
        sliding_window=None,
        use_sliding_window=False,
        max_window_layers=36,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_cache = use_cache
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.num_key_value_heads = num_key_value_heads

        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range

        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )