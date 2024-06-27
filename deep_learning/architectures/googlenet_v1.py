import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))


class InceptionBlock(nn.Module):
    """
    Building block of inception-v1 architecture consisting of 4 branches:
    1. branch1: 1x1 conv
    2. branch2: 1x1 conv -> 3x3 conv
    3. branch3: 1x1 conv -> 5x5 conv
    4. branch4: 3x3 maxpool -> 1x1 conv

    Note:
        1. Output and input feature map height and width should remain the same.
        To generate same height and width of output feature map, following padding is needed:
        - 1x1 conv: p=0
        - 3x3 conv: p=1
        - 5x5 conv: p=2
    """

    def __init__(self, in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_1x1_maxpool):
        """

        Args:
            in_channels: int - number of input channels
            out_1x1: int - number of output channels for branch 1
            in_3x3: int - number of output channels of 1x1 conv that are input to 3x3 conv
            out_3x3: int - number of output channels for branch 2
            in_5x5: int - number of output channels of 1x1 conv that are input to 5x5 conv
            out_5x5: int - number of output channels for branch 3
            out_1x1_maxpool: int - number of output channels for branch 4
        """

        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=1, stride=1, padding=0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_3x3, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=in_3x3, out_channels=out_3x3, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_5x5, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=in_5x5, out_channels=out_5x5, kernel_size=5, stride=1, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=out_1x1_maxpool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet_V1(nn.Module):
    def __init__(self):