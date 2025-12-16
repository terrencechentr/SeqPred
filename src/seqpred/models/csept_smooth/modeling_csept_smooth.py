from transformers import Qwen2PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

# Try to import can_return_tuple (not available in transformers 4.40.2)
try:
    from transformers.utils import can_return_tuple
except ImportError:
    def can_return_tuple(func):
        return func

# Try to import cache types
try:
    from transformers.models.qwen2.modeling_qwen2 import StaticCache, SlidingWindowCache, DynamicCache
except ImportError:
    try:
        from transformers.cache_utils import StaticCache, SlidingWindowCache, DynamicCache
    except ImportError:
        StaticCache = type(None)
        SlidingWindowCache = type(None)
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            DynamicCache = dict

# Try to import attention utilities (not used in simplified version)
AttentionMaskConverter = None
BlockMask = None
make_flex_block_causal_mask = None

from .configuration_csept_smooth import CseptSmoothConfig
import torch.nn.functional as F

logger = logging.get_logger(__name__)


@dataclass
class CseptSmoothOutput(ModelOutput):
    """
    输出类，包含平滑和恢复后的预测结果
    
    Args:
        loss: 预测损失
        preds: 恢复后的预测值（原始尺度）
        smoothed_input: 平滑后的输入（用于分析）
        hidden_states: Transformer隐藏状态
        attentions: 注意力权重
    """
    loss: Optional[torch.FloatTensor] = None
    preds: Optional[torch.FloatTensor] = None
    smoothed_input: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class SmoothingLayer(nn.Module):
    """
    平滑层：对输入数据进行平滑处理
    """
    
    def __init__(self, window_size: int, num_features: int, target_features: list, learnable: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.target_features = target_features
        self.learnable = learnable
        
        if learnable:
            weights = torch.ones(len(target_features), window_size) / window_size
            self.weights = nn.Parameter(weights)
        else:
            weights = torch.ones(len(target_features), window_size) / window_size
            self.register_buffer('weights', weights)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: (batch_size, seq_len, num_features)
        Returns:
            smoothed_x: (batch_size, seq_len, num_features)
        """
        batch_size, seq_len, num_features = x.shape
        output = x.clone()
        
        for idx, feat_idx in enumerate(self.target_features):
            feature = x[:, :, feat_idx]  # (batch_size, seq_len)
            
            if self.learnable:
                weights = F.softmax(self.weights[idx], dim=0)
            else:
                weights = self.weights[idx]
            
            # 使用1D卷积实现滚动平均
            feature_unsqueezed = feature.unsqueeze(1)  # (batch_size, 1, seq_len)
            kernel = weights.view(1, 1, -1)  # (1, 1, window_size)
            
            # 使用replicate padding保持序列长度
            padded_feature = F.pad(feature_unsqueezed, (self.window_size - 1, 0), mode='replicate')
            smoothed = F.conv1d(padded_feature, kernel)
            
            output[:, :, feat_idx] = smoothed.squeeze(1)
        
        return output


class UnsmoothingLayer(nn.Module):
    """
    恢复层：将平滑后的预测恢复到原始尺度
    
    实现思路：
    1. 如果平滑是y_smooth = avg(y_raw)，那么恢复就是反向操作
    2. 可以使用可学习的线性层或反卷积来实现
    """
    
    def __init__(self, window_size: int, num_features: int, target_features: list, mode: str = 'identity'):
        """
        Args:
            window_size: 平滑窗口大小
            num_features: 特征数量
            target_features: 需要恢复的特征索引
            mode: 恢复模式
                - 'identity': 直接返回（假设平滑后已经足够好）
                - 'learnable': 使用可学习的线性层恢复
                - 'inverse': 简单的反向平滑（加入高频成分）
        """
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.target_features = target_features
        self.mode = mode
        
        if mode == 'learnable':
            # 为每个目标特征创建可学习的恢复权重
            self.recovery_weights = nn.ParameterList([
                nn.Parameter(torch.ones(window_size) / window_size)
                for _ in target_features
            ])
            self.recovery_bias = nn.ParameterList([
                nn.Parameter(torch.zeros(1))
                for _ in target_features
            ])
    
    def forward(self, x_smooth: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x_smooth: 平滑后的数据 (batch_size, seq_len, num_features)
        Returns:
            x_recovered: 恢复后的数据 (batch_size, seq_len, num_features)
        """
        if self.mode == 'identity':
            # 直接返回平滑后的结果
            return x_smooth
        
        elif self.mode == 'learnable':
            # 使用可学习的权重进行恢复
            output = x_smooth.clone()
            
            for idx, feat_idx in enumerate(self.target_features):
                feature = x_smooth[:, :, feat_idx]
                
                # 应用可学习的反向操作
                feature_unsqueezed = feature.unsqueeze(1)
                kernel = self.recovery_weights[idx].view(1, 1, -1)
                
                # 使用转置卷积进行"反平滑"
                padded_feature = F.pad(feature_unsqueezed, (self.window_size - 1, 0), mode='replicate')
                recovered = F.conv1d(padded_feature, kernel)
                
                # 添加偏置
                recovered = recovered + self.recovery_bias[idx]
                
                output[:, :, feat_idx] = recovered.squeeze(1)
            
            return output
        
        elif self.mode == 'inverse':
            # 简单的反向平滑：添加高频成分
            output = x_smooth.clone()
            
            for feat_idx in self.target_features:
                # 计算一阶差分（高频成分）
                diff = x_smooth[:, 1:, feat_idx] - x_smooth[:, :-1, feat_idx]
                # 添加回一部分高频成分
                output[:, 1:, feat_idx] = x_smooth[:, 1:, feat_idx] + 0.3 * diff
            
            return output
        
        else:
            raise ValueError(f"Unknown unsmoothing mode: {self.mode}")


class CseptSmoothModel(Qwen2PreTrainedModel):
    """
    改进的 CSEPT-Smooth 模型
    
    架构流程：
    1. 输入原始数据
    2. 平滑层处理
    3. Embedding投影
    4. Transformer处理
    5. 输出层
    6. 恢复层（将平滑的预测恢复到原始尺度）
    """
    
    config_class = CseptSmoothConfig  # 指定配置类
    
    def __init__(self, config: CseptSmoothConfig):
        super().__init__(config)
        
        # 1. 平滑层（在embedding之前）
        self.smoothing = SmoothingLayer(
            window_size=config.smooth_window_size,
            num_features=config.input_size,
            target_features=config.smooth_target_features,
            learnable=config.smooth_learnable
        )
        
        # 2. Embedding层
        self.embed_proj = nn.Linear(config.input_size, config.hidden_size)
        
        # 3. Transformer层
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 4. RoPE
        try:
            self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        except TypeError:
            self.rotary_emb = Qwen2RotaryEmbedding(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
        
        self.gradient_checkpointing = False
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_smoothed: bool = False,  # 新参数：是否返回平滑后的输入
    ) -> BaseModelOutputWithPast:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_values is None and input_embeds is None:
            raise ValueError("必须指定 input_values 或 input_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` 与梯度检查点不兼容，设置为 False")
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("`past_key_values` 应该是 Cache 对象或 None")

        if input_embeds is None:
            # 步骤1: 平滑处理
            smoothed_values = self.smoothing(input_values)
            
            # 步骤2: Embedding投影
            inputs_embeds = self.embed_proj(smoothed_values)
        else:
            inputs_embeds = input_embeds
            smoothed_values = None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # 计算position embeddings for RoPE
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 步骤3: Transformer处理
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 如果需要，附加平滑后的输入用于分析
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
        if return_smoothed and smoothed_values is not None:
            output.smoothed_input = smoothed_values
        
        return output

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        # 简化版本，避免复杂的兼容性问题
        if attention_mask is not None:
            return attention_mask
        
        # 创建因果mask
        batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[1]
        
        # 创建因果mask: (seq_length, seq_length)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=input_tensor.device) * float('-inf'),
            diagonal=1
        )
        
        # 扩展到 (batch_size, 1, seq_length, seq_length)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        return causal_mask


class CspetSmoothForSequencePrediction(Qwen2PreTrainedModel):
    """
    完整的序列预测模型：平滑 → Transformer → 恢复
    """
    
    config_class = CseptSmoothConfig  # 指定配置类
    
    def __init__(self, config: CseptSmoothConfig):
        super().__init__(config)
        self.model = CseptSmoothModel(config)
        
        # 输出层：从hidden_size映射到1（预测值）
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        
        # 恢复层：将平滑的预测恢复到原始尺度
        self.unsmoothing = UnsmoothingLayer(
            window_size=config.smooth_window_size,
            num_features=1,  # 预测输出只有1个特征
            target_features=[0],
            mode=config.unsmooth_mode if hasattr(config, 'unsmooth_mode') else 'identity'
        )
        
        self.loss_type = config.loss_type
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional["Cache"] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_smoothed: bool = False,
        **kwargs
    ):
        # 步骤1-3: 平滑 → Transformer
        outputs = self.model(
            input_values=input_values,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_smoothed=return_smoothed,
        )
        
        # 步骤4: 输出层（得到平滑的预测）
        sequence_output_smooth = self.score(outputs.last_hidden_state)
        
        # 步骤5: 恢复层（恢复到原始尺度）
        sequence_output = self.unsmoothing(sequence_output_smooth)

        loss = None
        if labels is not None:
            loss = self.compute_loss(sequence_output, labels)

        return CseptSmoothOutput(
            loss=loss,
            preds=sequence_output,
            smoothed_input=getattr(outputs, 'smoothed_input', None) if return_smoothed else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_loss(self, preds: torch.FloatTensor, labels: torch.FloatTensor):
        """计算损失"""
        preds = preds[:, :-1, :]
        labels = labels[:, 1:, :]

        if self.loss_type == "mse":
            return F.mse_loss(preds, labels)
        elif self.loss_type == "mae":
            return F.l1_loss(preds, labels)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(preds, labels)
        elif self.loss_type == "log_cosh":
            diff = preds - labels
            return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
        elif self.loss_type == "huber":
            return F.huber_loss(preds, labels, delta=1.0)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

    @torch.no_grad()
    def generate(self, input_values: torch.FloatTensor, max_length: int, noise_level: float = 0.1):
        """自回归生成"""
        generated_values = [input_values.clone()]
        for _ in range(max_length):
            output = self.forward(input_values=input_values)
            preds = output.preds[:, -1:, :]
            input_values = torch.cat(generated_values + [preds], dim=-2)
            input_values = input_values + torch.randn_like(input_values) * noise_level
            generated_values.append(preds)
        return torch.cat(generated_values, dim=-2)


# 注册到AutoModel
try:
    from transformers import AutoModel
    AutoModel.register(CseptSmoothConfig, CspetSmoothForSequencePrediction)
except ImportError:
    pass
