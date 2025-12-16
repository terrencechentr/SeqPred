"""
改进版CSEPT-Smooth模型 - 添加方向损失
解决方向准确率低的问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.csept_smooth.configuration_csept_smooth import CseptSmoothConfig
from models.csept_smooth.modeling_csept_smooth import (
    CspetSmoothForSequencePrediction, 
    CseptSmoothOutput
)


class CspetSmoothWithDirectionLoss(CspetSmoothForSequencePrediction):
    """
    改进版模型：添加方向损失
    
    改进点:
    1. 添加方向BCE损失
    2. 可配置的损失权重
    3. 输出方向预测概率
    """
    
    def __init__(self, config: CseptSmoothConfig, direction_loss_weight=0.3):
        super().__init__(config)
        self.direction_loss_weight = direction_loss_weight
        print(f"✓ Initialized with direction loss weight: {direction_loss_weight}")
    
    def compute_loss(self, preds: torch.FloatTensor, labels: torch.FloatTensor):
        """
        计算组合损失 = 幅度损失 + 方向损失
        
        Args:
            preds: 预测值 (batch, seq_len, 1)
            labels: 真实值 (batch, seq_len, 1)
        
        Returns:
            combined_loss: 组合损失
        """
        # Teacher forcing: 预测t时刻，标签为t+1时刻
        preds = preds[:, :-1, :]
        labels = labels[:, 1:, :]
        
        # 1. 幅度损失 (MSE)
        if self.loss_type == "mse":
            magnitude_loss = F.mse_loss(preds, labels)
        elif self.loss_type == "mae":
            magnitude_loss = F.l1_loss(preds, labels)
        elif self.loss_type == "smooth_l1":
            magnitude_loss = F.smooth_l1_loss(preds, labels)
        elif self.loss_type == "huber":
            magnitude_loss = F.huber_loss(preds, labels, delta=1.0)
        else:
            magnitude_loss = F.mse_loss(preds, labels)
        
        # 2. 方向损失 (Binary Cross Entropy)
        # 真实方向: 1=上涨(>0), 0=下跌(<=0)
        true_direction = (labels > 0).float()
        
        # 预测方向概率: 使用sigmoid将预测值转换为[0,1]
        # 放大信号以增强方向敏感性
        pred_direction_prob = torch.sigmoid(preds * 10.0)
        
        # BCE损失
        direction_loss = F.binary_cross_entropy(
            pred_direction_prob,
            true_direction,
            reduction='mean'
        )
        
        # 3. 组合损失
        alpha = self.direction_loss_weight
        combined_loss = (1 - alpha) * magnitude_loss + alpha * direction_loss
        
        # 用于调试
        if torch.isnan(combined_loss):
            print(f"⚠️  NaN detected!")
            print(f"  magnitude_loss: {magnitude_loss.item()}")
            print(f"  direction_loss: {direction_loss.item()}")
        
        return combined_loss
    
    def compute_direction_accuracy(self, preds: torch.FloatTensor, labels: torch.FloatTensor):
        """
        计算方向准确率
        
        Args:
            preds: 预测值
            labels: 真实值
        
        Returns:
            accuracy: 方向准确率 (0-1)
        """
        preds = preds[:, :-1, :]
        labels = labels[:, 1:, :]
        
        pred_direction = (preds > 0).float()
        true_direction = (labels > 0).float()
        
        accuracy = (pred_direction == true_direction).float().mean()
        return accuracy.item()


class CspetSmoothMultiTask(CspetSmoothForSequencePrediction):
    """
    多任务学习版本：同时学习回归和分类
    
    改进点:
    1. 添加独立的方向分类头
    2. 多任务联合训练
    3. 可以输出方向概率和置信度
    """
    
    def __init__(self, config: CseptSmoothConfig, direction_loss_weight=0.3):
        super().__init__(config)
        self.direction_loss_weight = direction_loss_weight
        
        # 添加方向分类头
        self.direction_head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # 2类：上涨(1) / 下跌(0)
        )
        
        print(f"✓ Initialized multi-task model with direction classification head")
    
    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        """
        前向传播
        
        Returns:
            CseptSmoothOutput with additional direction_logits
        """
        # 获取transformer输出
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 回归预测（幅度）
        sequence_output_smooth = self.score(outputs.last_hidden_state)
        sequence_output = self.unsmoothing(sequence_output_smooth)
        
        # 分类预测（方向）
        direction_logits = self.direction_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            # 1. 回归损失
            regression_loss = self.compute_regression_loss(sequence_output, labels)
            
            # 2. 分类损失
            classification_loss = self.compute_classification_loss(
                direction_logits, 
                labels
            )
            
            # 3. 组合损失
            alpha = self.direction_loss_weight
            loss = (1 - alpha) * regression_loss + alpha * classification_loss
        
        return {
            'loss': loss,
            'preds': sequence_output,
            'direction_logits': direction_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
    
    def compute_regression_loss(self, preds: torch.FloatTensor, labels: torch.FloatTensor):
        """计算回归损失"""
        preds = preds[:, :-1, :]
        labels = labels[:, 1:, :]
        
        if self.loss_type == "mse":
            return F.mse_loss(preds, labels)
        elif self.loss_type == "mae":
            return F.l1_loss(preds, labels)
        else:
            return F.mse_loss(preds, labels)
    
    def compute_classification_loss(
        self, 
        direction_logits: torch.FloatTensor, 
        labels: torch.FloatTensor
    ):
        """计算分类损失"""
        # Teacher forcing
        direction_logits = direction_logits[:, :-1, :]  # (batch, seq-1, 2)
        labels = labels[:, 1:, :]  # (batch, seq-1, 1)
        
        # 方向标签: 1=上涨, 0=下跌
        direction_labels = (labels > 0).long().squeeze(-1)  # (batch, seq-1)
        
        # Cross Entropy损失
        batch_size, seq_len, num_classes = direction_logits.shape
        direction_logits_flat = direction_logits.reshape(-1, num_classes)
        direction_labels_flat = direction_labels.reshape(-1)
        
        classification_loss = F.cross_entropy(
            direction_logits_flat,
            direction_labels_flat,
            reduction='mean'
        )
        
        return classification_loss
    
    def compute_direction_accuracy(self, direction_logits: torch.FloatTensor, labels: torch.FloatTensor):
        """计算方向准确率"""
        direction_logits = direction_logits[:, :-1, :]
        labels = labels[:, 1:, :]
        
        # 预测类别
        pred_classes = torch.argmax(direction_logits, dim=-1)
        
        # 真实类别
        true_classes = (labels > 0).long().squeeze(-1)
        
        # 准确率
        accuracy = (pred_classes == true_classes).float().mean()
        return accuracy.item()
    
    def predict_with_direction(self, input_values: torch.FloatTensor):
        """
        预测同时返回幅度和方向
        
        Returns:
            preds: 预测幅度
            direction_probs: 方向概率 (上涨的概率)
            direction_confidence: 方向置信度
        """
        with torch.no_grad():
            outputs = self.forward(input_values=input_values)
            
            # 幅度预测
            preds = outputs.preds
            
            # 方向概率
            direction_probs = F.softmax(outputs.direction_logits, dim=-1)
            up_prob = direction_probs[:, :, 1]  # 上涨概率
            
            # 置信度（最大概率与0.5的距离）
            confidence = torch.abs(up_prob - 0.5) * 2
            
            return preds, up_prob, confidence


# 方便导入
__all__ = [
    'CspetSmoothWithDirectionLoss',
    'CspetSmoothMultiTask',
]

# 注册到AutoModel
try:
    from transformers import AutoModel
    AutoModel.register(CseptSmoothConfig, CspetSmoothWithDirectionLoss)
    AutoModel.register(CseptSmoothConfig, CspetSmoothMultiTask)
except ImportError:
    pass

