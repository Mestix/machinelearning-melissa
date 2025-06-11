"""
Recurrent Neural Network (RNN) implementations for sequence classification.

This module provides several RNN architectures:
1. RecurrentNeuralNetworkWithGRU: GRU-based RNN
2. RecurrentNeuralNetwork: Basic RNN with tanh activation
3. GRUWithAttention: GRU-based RNN with attention mechanism
4. RecurrentNeuralNetworkWithAttention: Basic RNN with attention mechanism

All models are designed for sequence classification tasks, particularly for the Gestures dataset.
The module also includes a ModelConfig dataclass for easy configuration of RNN parameters.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration class for RNN models.

    This dataclass holds the configuration parameters for all RNN models,
    making it easy to create and configure different RNN architectures.

    Attributes:
        input_size (int): Size of input features at each time step
        hidden_size (int): Size of hidden state in the RNN
        num_layers (int): Number of recurrent layers
        output_size (int): Number of output classes
        dropout (float): Dropout rate for regularization (default: 0.0)
    """

    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.0


class RecurrentNeuralNetworkWithGRU(nn.Module):
    """
    GRU-based Recurrent Neural Network for sequence classification.

    This model uses Gated Recurrent Unit (GRU) cells, which are more efficient
    and often perform better than standard RNN cells. It includes layer normalization
    after the RNN layer and uses the last time step for classification.

    Attributes:
        config (ModelConfig): Configuration parameters
        rnn (nn.GRU): GRU layer
        norm (nn.LayerNorm): Layer normalization
        linear (nn.Linear): Output linear layer for classification
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the GRU-based RNN model.

        Args:
            config (ModelConfig): Configuration parameters for the model
        """
        super().__init__()
        self.config = config

        # GRU layer
        self.rnn = nn.GRU(
            input_size=config.input_size,  # Size of input features at each time step
            hidden_size=config.hidden_size,  # Size of hidden state
            dropout=config.dropout,  # Dropout rate for regularization
            batch_first=True,  # Input shape is [batch, seq_len, features]
            num_layers=config.num_layers,  # Number of recurrent layers
        )

        # Layer normalization (normalizes per time step and per sample across features)
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output linear layer for classification
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, output_size]
        """
        x, _ = self.rnn(
            x
        )  # Process sequence through GRU, x: [batch, seq_len, hidden_size]
        x = self.norm(x)  # Apply layer normalization over hidden dimension
        last_step = x[:, -1, :]  # Take the last time step output
        yhat = self.linear(last_step)  # Apply linear layer for classification
        return yhat


class RecurrentNeuralNetwork(nn.Module):
    """
    Basic Recurrent Neural Network for sequence classification.

    This model uses standard RNN cells with tanh activation. It includes layer
    normalization after the RNN layer and uses the last time step for classification.

    Attributes:
        config (ModelConfig): Configuration parameters
        rnn (nn.RNN): RNN layer with tanh activation
        norm (nn.LayerNorm): Layer normalization
        linear (nn.Linear): Output linear layer for classification
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the basic RNN model.

        Args:
            config (ModelConfig): Configuration parameters for the model
        """
        super().__init__()
        self.config = config

        # Basic RNN layer
        self.rnn = nn.RNN(
            input_size=config.input_size,  # Size of input features at each time step
            hidden_size=config.hidden_size,  # Size of hidden state
            num_layers=config.num_layers,  # Number of recurrent layers
            dropout=(
                config.dropout if config.num_layers > 1 else 0.0
            ),  # Dropout only applied between layers
            batch_first=True,  # Input shape is [batch, seq_len, features]
            nonlinearity="tanh",  # Activation function (default)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output linear layer for classification
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, output_size]
        """
        x, _ = self.rnn(
            x
        )  # Process sequence through RNN, x: [batch, seq_len, hidden_size]
        x = self.norm(x)  # Apply layer normalization over hidden dimension
        last_step = x[:, -1, :]  # Take the last time step output
        yhat = self.linear(last_step)  # Apply linear layer for classification
        return yhat


class AttentionLayer(nn.Module):
    """
    Attention mechanism for focusing on important time steps in a sequence.

    This layer computes attention scores for each time step, normalizes them
    using softmax to get attention weights, and computes a weighted sum of
    the RNN outputs to create a context vector.

    Attributes:
        attn (nn.Linear): Linear layer to compute attention scores
    """

    def __init__(self, hidden_size: int):
        """
        Initialize the attention layer.

        Args:
            hidden_size (int): Size of the hidden state from the RNN
        """
        super().__init__()
        # Linear layer to compute attention scores
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs: Tensor) -> Tensor:
        """
        Forward pass through the attention layer.

        Args:
            rnn_outputs (Tensor): RNN outputs of shape [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: Context vector of shape [batch_size, hidden_size]
        """
        # Compute attention scores
        attn_scores = self.attn(rnn_outputs)  # Shape: [batch, seq_len, 1]

        # Normalize scores to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=1)  # Softmax over time dimension

        # Compute weighted sum to get context vector
        context = torch.sum(
            attn_weights * rnn_outputs, dim=1
        )  # Shape: [batch, hidden_size]
        return context


class GRUWithAttention(nn.Module):
    """
    GRU-based Recurrent Neural Network with attention mechanism.

    This model extends the basic GRU model by adding an attention mechanism
    to focus on important time steps in the sequence. It includes layer
    normalization after the RNN layer.

    Attributes:
        rnn (nn.GRU): GRU layer
        attention (AttentionLayer): Attention mechanism
        norm (nn.LayerNorm): Layer normalization
        fc (nn.Linear): Output linear layer for classification
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the GRU with attention model.

        Args:
            config (ModelConfig): Configuration parameters for the model
        """
        super().__init__()

        # GRU layer
        self.rnn = nn.GRU(
            input_size=config.input_size,  # Size of input features at each time step
            hidden_size=config.hidden_size,  # Size of hidden state
            num_layers=config.num_layers,  # Number of recurrent layers
            dropout=(
                config.dropout if config.num_layers > 1 else 0.0
            ),  # Dropout only applied between layers
            batch_first=True,  # Input shape is [batch, seq_len, features]
        )

        # Attention mechanism
        self.attention = AttentionLayer(config.hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output linear layer for classification
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, output_size]
        """
        rnn_out, _ = self.rnn(
            x
        )  # Process sequence through GRU, shape: [batch, seq_len, hidden]
        rnn_out = self.norm(rnn_out)  # Apply layer normalization
        context = self.attention(
            rnn_out
        )  # Apply attention to get context vector, shape: [batch, hidden]
        output = self.fc(
            context
        )  # Apply linear layer for classification, shape: [batch, num_classes]
        return output


class RecurrentNeuralNetworkWithAttention(nn.Module):
    """
    Basic Recurrent Neural Network with attention mechanism.

    This model extends the basic RNN model by adding an attention mechanism
    to focus on important time steps in the sequence. It includes layer
    normalization after the RNN layer.

    Attributes:
        config (ModelConfig): Configuration parameters
        rnn (nn.RNN): RNN layer with tanh activation
        attention (AttentionLayer): Attention mechanism
        norm (nn.LayerNorm): Layer normalization
        linear (nn.Linear): Output linear layer for classification
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the RNN with attention model.

        Args:
            config (ModelConfig): Configuration parameters for the model
        """
        super().__init__()
        self.config = config

        # Basic RNN layer
        self.rnn = nn.RNN(
            input_size=config.input_size,  # Size of input features at each time step
            hidden_size=config.hidden_size,  # Size of hidden state
            num_layers=config.num_layers,  # Number of recurrent layers
            dropout=(
                config.dropout if config.num_layers > 1 else 0.0
            ),  # Dropout only applied between layers
            batch_first=True,  # Input shape is [batch, seq_len, features]
            nonlinearity="tanh",  # Activation function
        )

        # Attention mechanism
        self.attention = AttentionLayer(config.hidden_size)

        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)

        # Output linear layer for classification
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, input_size]

        Returns:
            Tensor: Output tensor of shape [batch_size, output_size]
        """
        rnn_out, _ = self.rnn(
            x
        )  # Process sequence through RNN, shape: [batch, seq_len, hidden_size]
        rnn_out = self.norm(rnn_out)  # Apply layer normalization per time step
        context = self.attention(rnn_out)  # Apply attention over all time steps
        yhat = self.linear(context)  # Apply linear layer for classification
        return yhat
