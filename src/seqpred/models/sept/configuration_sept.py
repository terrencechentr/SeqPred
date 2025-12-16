# coding=utf-8
"""Sept model configuration - 基于Qwen2的序列预测模型配置"""

from transformers import Qwen2Config
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SeptConfig(Qwen2Config):
    """
    Sept模型配置，继承自Qwen2Config
    
    Sept使用标准的Qwen2模型结构进行序列预测任务
    """
    
    model_type = "sept"
    
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        classifier_dropout=None,
        **kwargs,
    ):
        self.classifier_dropout = classifier_dropout
        
        # 兼容性：添加layer_types（新版transformers需要）
        if 'layer_types' not in kwargs:
            kwargs['layer_types'] = ["full_attention"] * num_hidden_layers
        
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            **kwargs,
        )


__all__ = ["SeptConfig"]

# 注册到AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("sept", SeptConfig)
except ImportError:
    pass

