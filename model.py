from torch import nn


class DummyModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)
