#!/usr/bin/env python3
"""
Naive RNN for next-step prediction.
- Input: (batch, seq_len, features)
- Loss: align preds[:, :-1] with labels[:, 1:]
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_rnn_naive import RnnNaiveConfig


def _get_loss_fn(loss_type: str):
    loss_type = (loss_type or "mse").lower()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type in ["smooth_l1", "huber"]:
        return nn.SmoothL1Loss()
    return nn.MSELoss()


class RnnNaiveModel(PreTrainedModel):
    config_class = RnnNaiveConfig

    def __init__(self, config: RnnNaiveConfig):
        super().__init__(config)
        self.rnn = nn.RNN(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_size, 1)
        self.loss_fn = _get_loss_fn(config.loss_type)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Union[dict, Tuple[torch.Tensor, ...]]:
        outputs, hidden = self.rnn(input_values, hidden_state)
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


__all__ = ["RnnNaiveModel"]

# Register with AutoModel/AutoConfig
try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("rnn_naive", RnnNaiveConfig)
    AutoModel.register(RnnNaiveConfig, RnnNaiveModel)
except Exception:
    pass

