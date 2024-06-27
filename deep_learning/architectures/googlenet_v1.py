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
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(
            in_channels=192,
            out_1x1=64,
            in_3x3=96, out_3x3=128,
            in_5x5=16, out_5x5=32,
            out_1x1_maxpool=32
        )
        self.inception3b = InceptionBlock(
            in_channels=256,
            out_1x1=128,
            in_3x3=128, out_3x3=192,
            in_5x5=32, out_5x5=96,
            out_1x1_maxpool=64
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(
            in_channels=480,
            out_1x1=192,
            in_3x3=96, out_3x3=208,
            in_5x5=16, out_5x5=48,
            out_1x1_maxpool=64
        )
        self.inception4b = InceptionBlock(
            in_channels=512,
            out_1x1=160,
            in_3x3=112, out_3x3=224,
            in_5x5=24, out_5x5=64,
            out_1x1_maxpool=64
        )
        self.inception4c = InceptionBlock(
            in_channels=512,
            out_1x1=128,
            in_3x3=128, out_3x3=256,
            in_5x5=24, out_5x5=64,
            out_1x1_maxpool=64
        )
        self.inception4d = InceptionBlock(
            in_channels=512,
            out_1x1=112,
            in_3x3=144, out_3x3=288,
            in_5x5=32, out_5x5=64,
            out_1x1_maxpool=64
        )
        self.inception4e = InceptionBlock(
            in_channels=528,
            out_1x1=256,
            in_3x3=160, out_3x3=320,
            in_5x5=32, out_5x5=128,
            out_1x1_maxpool=128
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(
            in_channels=832,
            out_1x1=256,
            in_3x3=160, out_3x3=320,
            in_5x5=32, out_5x5=128,
            out_1x1_maxpool=128
        )
        self.inception5b = InceptionBlock(
            in_channels=832,
            out_1x1=384,
            in_3x3=192, out_3x3=384,
            in_5x5=48, out_5x5=128,
            out_1x1_maxpool=128
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x
