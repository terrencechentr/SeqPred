# coding=utf-8
"""Csept Smooth model configuration - 带有平滑和恢复机制的配置"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CseptSmoothConfig(PretrainedConfig):
    """
    CSEPT-Smooth模型配置
    
    新增参数:
        smooth_window_size: 平滑窗口大小
        smooth_learnable: 平滑权重是否可学习
        smooth_target_features: 需要平滑的特征索引列表
        unsmooth_mode: 恢复模式 ('identity', 'learnable', 'inverse')
    """

    model_type = "csept_smooth"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        input_size=5,
        smooth_window_size=50,
        smooth_learnable=False,
        smooth_target_features=None,
        unsmooth_mode='identity',
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
        loss_type="mse",
        **kwargs,
    ):
        self.input_size = input_size
        self.smooth_window_size = smooth_window_size
        self.smooth_learnable = smooth_learnable
        self.smooth_target_features = smooth_target_features if smooth_target_features is not None else [0]
        self.unsmooth_mode = unsmooth_mode
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.loss_type = loss_type
        
        # 兼容性参数
        if 'layer_types' not in kwargs:
            self.layer_types = ["full_attention"] * num_hidden_layers
        
        # 向后兼容
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        
        # 计算head_dim
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # RoPE参数（用于兼容新版transformers）
        self.rope_parameters = {
            "rope_type": "default",
            "factor": 1.0,
            "dim": self.head_dim,
        }

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["CseptSmoothConfig"]

# 注册到AutoConfig
try:
    from transformers import AutoConfig
    AutoConfig.register("csept_smooth", CseptSmoothConfig)
except ImportError:
    pass
