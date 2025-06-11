"""
Neural Network implementations for image classification.

This module provides two neural network architectures:
1. NeuralNetwork: A simple neural network with 2 hidden layers
2. DeepNeuralNetwork: A deeper neural network with 3 hidden layers

Both are designed for the Fashion MNIST dataset with configurable layer sizes.
"""

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    """
    A simple neural network with 2 hidden layers for image classification.
    
    This network flattens the input image and passes it through a stack of
    fully connected layers with ReLU activations. It's designed for the
    Fashion MNIST dataset with 28x28 pixel images.
    
    Attributes:
        flatten (nn.Flatten): Layer to flatten the input image
        linear_relu_stack (nn.Sequential): Stack of linear layers with ReLU activations
    """
    def __init__(self, num_classes: int, units1: int, units2: int) -> None:
        """
        Initialize the neural network.
        
        Args:
            num_classes (int): Number of output classes (default: 10 for Fashion MNIST)
            units1 (int): Number of units in the first hidden layer
            units2 (int): Number of units in the second hidden layer
        """
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten the 28x28 input images

        # Stack of linear layers with ReLU activations
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, units1),     # Input layer: 784 (28x28) -> units1
            nn.ReLU(),                      # Non-linear activation
            nn.Linear(units1, units2),      # First hidden layer: units1 -> units2
            nn.ReLU(),                      # Non-linear activation
            nn.Linear(units2, num_classes), # Output layer: units2 -> num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = self.flatten(x)                # Flatten input from [batch, 1, 28, 28] to [batch, 784]
        logits = self.linear_relu_stack(x) # Pass through the linear layers with ReLU activations
        return logits


class DeepNeuralNetwork(nn.Module):
    """
    A deeper neural network with 3 hidden layers for image classification.
    
    This network extends the basic NeuralNetwork by adding an additional hidden layer,
    providing more capacity for learning complex patterns. It's designed for the
    Fashion MNIST dataset with 28x28 pixel images.
    
    Attributes:
        flatten (nn.Flatten): Layer to flatten the input image
        linear_relu_stack (nn.Sequential): Stack of linear layers with ReLU activations
    """
    def __init__(self, num_classes: int, units1: int, units2: int, units3: int) -> None:
        """
        Initialize the deep neural network.
        
        Args:
            num_classes (int): Number of output classes (default: 10 for Fashion MNIST)
            units1 (int): Number of units in the first hidden layer
            units2 (int): Number of units in the second hidden layer
            units3 (int): Number of units in the third hidden layer
        """
        super().__init__()
        self.flatten = nn.Flatten()  # Flatten the 28x28 input images

        # Stack of linear layers with ReLU activations
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, units1),     # Input layer: 784 (28x28) -> units1
            nn.ReLU(),                      # Non-linear activation
            nn.Linear(units1, units2),      # First hidden layer: units1 -> units2
            nn.ReLU(),                      # Non-linear activation
            nn.Linear(units2, units3),      # Second hidden layer: units2 -> units3
            nn.ReLU(),                      # Non-linear activation
            nn.Linear(units3, num_classes), # Output layer: units3 -> num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = self.flatten(x)                # Flatten input from [batch, 1, 28, 28] to [batch, 784]
        logits = self.linear_relu_stack(x) # Pass through the linear layers with ReLU activations
        return logits
