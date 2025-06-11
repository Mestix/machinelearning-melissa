"""
Convolutional Neural Network (CNN) implementation for image classification.

This module provides a CNN architecture optimized for the Fashion MNIST dataset,
with configurable parameters for filters, units, and dropout rate.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification.

    This CNN architecture consists of a convolutional block followed by fully connected layers.
    The convolutional block includes Conv2D, BatchNorm, ReLU, MaxPooling, and Dropout layers.
    The network automatically determines the output dimensions of the convolutional block
    to properly size the input to the fully connected layers.

    Attributes:
        conv (nn.Sequential): Convolutional block with Conv2D, BatchNorm, ReLU, MaxPool, and Dropout
        flatten_dim (int): Automatically calculated dimension after flattening conv output
        fc (nn.Sequential): Fully connected layers with ReLU activation
    """

    def __init__(
        self,
        filters: int = 32,
        units: int = 128,
        input_size=(1, 1, 28, 28),
        dropout=0.1,
        num_classes=10,
    ):
        """
        Initialize the CNN model.

        Args:
            filters (int): Number of filters in the convolutional layer (default: 32)
            units (int): Number of units in the hidden dense layer (default: 128)
            input_size (tuple): Size of the input images (default: (1, 1, 28, 28) for Fashion MNIST)
            dropout (float): Dropout rate for regularization (default: 0.1)
            num_classes (int): Number of output classes (default: 10 for Fashion MNIST)
        """
        super().__init__()

        # Convolutional block
        self.conv = nn.Sequential(
            nn.Conv2d(
                1, filters, kernel_size=3, padding=1
            ),  # Maintains 28x28 spatial dimensions
            nn.BatchNorm2d(filters),  # Normalizes activations for stable training
            nn.ReLU(),  # Non-linear activation
            nn.MaxPool2d(2),  # Reduces spatial dimensions to 14x14
            nn.Dropout(dropout),  # Prevents overfitting
        )

        # Automatically determine output dimension after convolution
        with torch.no_grad():
            dummy = torch.zeros(input_size)
            out = self.conv(dummy)
            self.flatten_dim = out.view(out.size(0), -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, units),  # Hidden layer
            nn.ReLU(),  # Non-linear activation
            nn.Linear(units, num_classes),  # Output layer
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        x = self.conv(x)  # Apply convolutional block
        x = x.view(x.size(0), -1)  # Flatten the output
        return self.fc(x)  # Apply fully connected layers
