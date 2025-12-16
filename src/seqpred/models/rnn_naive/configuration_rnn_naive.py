#!/usr/bin/env python3
from transformers import PretrainedConfig


class RnnNaiveConfig(PretrainedConfig):
    model_type = "rnn_naive"

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        loss_type: str = "mse",
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss_type = loss_type
        super().__init__(**kwargs)


__all__ = ["RnnNaiveConfig"]

