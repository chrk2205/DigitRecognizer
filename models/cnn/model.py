import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Flatten(),
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.LazyLinear(num_classes),
        )

    def forward(self, x):
        return self.model(x)
