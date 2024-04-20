import torch
from torch import nn


class ResNet18(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer2 = nn.Sequential(
            BasicBlock(32, 64, stride=2, downsample=self._downsample(32, 64)),
            BasicBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=self._downsample(64, 128)),
            BasicBlock(128, 128),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=self._downsample(128, 256)),
            BasicBlock(256, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.15)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _downsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            identity = self.downsample(x)
        output += identity
        output = self.relu(output)
        return output
