import torch.nn as nn

class MNISTConvClassifier(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_channels: int = 32, num_classes: int = 10, hidden_dim: int = 25):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )
    def forward(self, x):
        y = self.model(x)
        # print(y)
        return y