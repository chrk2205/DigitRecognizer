import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        print(logits)
        return logits



class MNISTConvClassifier(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_classes: int = 10, hidden_dim: int = 25):
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
        print(y)
        return y