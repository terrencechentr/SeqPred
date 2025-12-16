#!/usr/bin/env python3
from transformers import PretrainedConfig


class LstmNaiveConfig(PretrainedConfig):
    model_type = "lstm_naive"

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        loss_type: str = "mse",
        bidirectional: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss_type = loss_type
        self.bidirectional = bidirectional
        self.dropout = dropout
        super().__init__(**kwargs)


__all__ = ["LstmNaiveConfig"]

