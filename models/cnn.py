import torch.nn as nn
from dataclasses import asdict

from dataclasses import dataclass


# Config class voor je CNN
@dataclass
class CNNConfig:
    matrixshape: tuple = (28, 28)
    batchsize: int = 64
    input_channels: int = 1
    hidden: int = 64
    kernel_size: int = 3
    maxpool: int = 2
    num_layers: int = 4
    num_classes: int = 10
    dropout: float = 0.3


# Blokje met conv, batchnorm, relu, optioneel pooling en dropout
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool=False, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# CNN model met dynamisch aantal lagen
class CNNblocks(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = asdict(config)
        input_channels = config.input_channels
        kernel_size = config.kernel_size
        hidden = config.hidden
        pool_size = config.maxpool
        dropout = getattr(config, "dropout", 0.2)

        self.convolutions = nn.ModuleList()

        # Eerste laag
        self.convolutions.append(
            ConvBlock(input_channels, hidden, kernel_size, pool=True, dropout=dropout)
        )

        # Verdere lagen
        for i in range(1, config.num_layers):
            use_pool = i % 2 == 0
            self.convolutions.append(
                ConvBlock(hidden, hidden, kernel_size, pool=use_pool, dropout=dropout)
            )

        # Output dimensie berekenen
        num_pools = (config.num_layers + 1) // 2
        matrix_size = (config.matrixshape[0] // (pool_size**num_pools)) * (
            config.matrixshape[1] // (pool_size**num_pools)
        )
        flatten_size = matrix_size * hidden
        print(f"Calculated matrix size: {matrix_size}")
        print(f"Calculated flatten size: {flatten_size}")

        # Dense layers
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, config.num_classes),
        )

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        x = self.dense(x)
        return x
