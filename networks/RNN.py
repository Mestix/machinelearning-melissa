import torch
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


class RecurrentNeuralNetworkWithGRU(nn.Module):
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
    

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.rnn = nn.RNN(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            nonlinearity="tanh",  # default
        )

        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)                  # [batch, seq_len, hidden_size]
        x = self.norm(x)                    # LayerNorm over laatste dim
        last_step = x[:, -1, :]             # laatste tijdstap
        yhat = self.linear(last_step)       # classificatie-output
        return yhat


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs: Tensor) -> Tensor:
        # rnn_outputs: [batch, seq_len, hidden_size]
        attn_scores = self.attn(rnn_outputs)            # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # over tijd
        context = torch.sum(attn_weights * rnn_outputs, dim=1)  # [batch, hidden_size]
        return context
    
class GRUWithAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = AttentionLayer(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        rnn_out, _ = self.rnn(x)                        # [batch, seq_len, hidden]
        rnn_out = self.norm(rnn_out)
        context = self.attention(rnn_out)               # [batch, hidden]
        output = self.fc(context)                       # [batch, num_classes]
        return output
    
class RecurrentNeuralNetworkWithAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.rnn = nn.RNN(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            nonlinearity="tanh",
        )

        self.attention = AttentionLayer(config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        rnn_out, _ = self.rnn(x)                  # [batch, seq_len, hidden_size]
        rnn_out = self.norm(rnn_out)              # Normaliseer per tijdstap
        context = self.attention(rnn_out)         # Attention over alle tijdstappen
        yhat = self.linear(context)               # Classificatie-output
        return yhat