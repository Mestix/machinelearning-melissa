import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.0


class GRUmodel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.num_layers,
        )

        self.norm = nn.LayerNorm(
            config.hidden_size
        )  # Normaliseert per tijdstap en per sample over de features.
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)  # x: [batch, seq_len, hidden_size]
        x = self.norm(x)  # LayerNorm over laatste dim (hidden_size)
        last_step = x[:, -1, :]  # neem laatste tijdstap
        yhat = self.linear(last_step)  # classificatie-output
        return yhat
