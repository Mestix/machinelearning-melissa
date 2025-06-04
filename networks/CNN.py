import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        filters: int = 32,
        units: int = 128,
        input_size=(1, 1, 28, 28),
        dropout=0.1,
        num_classes=10,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, padding=1),  # 28x28
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Dropout(dropout),
        )

        # Automatisch outputdimensie bepalen
        with torch.no_grad():
            dummy = torch.zeros(input_size)
            out = self.conv(dummy)
            self.flatten_dim = out.view(out.size(0), -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, units), nn.ReLU(), nn.Linear(units, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
