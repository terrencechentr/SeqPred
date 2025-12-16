#!/usr/bin/env python3
"""
Naive LSTM for next-step prediction.
- Input: (batch, seq_len, features)
- Loss: align preds[:, :-1] with labels[:, 1:]
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_lstm_naive import LstmNaiveConfig


def _get_loss_fn(loss_type: str):
    loss_type = (loss_type or "mse").lower()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type in ["smooth_l1", "huber"]:
        return nn.SmoothL1Loss()
    return nn.MSELoss()


class LstmNaiveModel(PreTrainedModel):
    config_class = LstmNaiveConfig

    def __init__(self, config: LstmNaiveConfig):
        super().__init__(config)
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        direction_multiplier = 2 if config.bidirectional else 1
        self.head = nn.Linear(config.hidden_size * direction_multiplier, 1)
        self.loss_fn = _get_loss_fn(config.loss_type)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[dict, Tuple[torch.Tensor, ...]]:
        outputs, hidden = self.lstm(input_values, hidden_state)
        preds = self.head(outputs)  # (batch, seq_len, 1)

        loss = None
        if labels is not None:
            if labels.shape[1] < 2:
                raise ValueError("Sequence length must be >=2 for next-step loss")
            loss = self.loss_fn(preds[:, :-1, :], labels[:, 1:, :])

        return {
            "loss": loss,
            "preds": preds,
            "logits": preds,
            "hidden_state": hidden,
        }


__all__ = ["LstmNaiveModel"]

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("lstm_naive", LstmNaiveConfig)
    AutoModel.register(LstmNaiveConfig, LstmNaiveModel)
except Exception:
    pass

