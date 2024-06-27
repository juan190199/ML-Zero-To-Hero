import torch
import torch.nn as nn

# resnetX = (number of channels, repetition, bottleneck expansion , bottleneck layer)
model_hparams = {}
model_hparams['resnet18'] = ([64, 128, 256, 512], [2, 2, 2, 2], 1, False)
model_hparams['resnet34'] = ([64, 128, 256, 512], [3, 4, 6, 3], 1, False)
model_hparams['resnet50'] = ([64, 128, 256, 512], [3, 4, 6, 3], 4, True)
model_hparams['resnet101'] = ([64, 128, 256, 512], [3, 4, 23, 3], 4, True)
model_hparams['resnet152'] = ([64, 128, 256, 512], [3, 8, 36, 3], 4, True)


class Bottleneck(nn.Module):
    """
    Creates a bottleneck building block with conv 1x1 -> 3x3 -> 1x1 layers
    """

    def __init__(self, in_channels, inter_channels, expansion, is_bottleneck, stride):
        """
        Args:
            in_channels: int - number of input channels to the bottleneck block
            inter_channels: int - number of channels to 3x3 conv
            expansion: int - factor by which the input channels are increased
            is_bottleneck: boolean - if is_bottleneck == False (3x3 -> 3x3), else (1x1 -> 3x3 -> 1x1)
            stride: int - stride applied in the 3x3 conv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.expansion = expansion
        self.is_bottleneck = is_bottleneck

        # if dim(x) == dim(F), use identity function
        if self.in_channels == self.inter_channels * self.expansion:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.inter_channels * self.expansion,
                    kernel_size=1, stride=stride, padding=0, bias=False
                )
            )
            projection_layer.append(nn.BatchNorm2d(self.inter_channels * self.expansion))
            # Only conv -> BN. No ReLU
            self.projection = nn.Sequential(*projection_layer)

        self.relu = nn.ReLU()

        if self.is_bottleneck:
            # Bottleneck
            # 1x1 conv
            self.conv1_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.batchnorm1 = nn.BatchNorm2d(self.inter_channels)

            # 3x3 conv
            self.conv2_3x3 = nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=self.inter_channels,
                kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.batchnorm2 = nn.BatchNorm2d(self.inter_channels)

            # 1x1 conv
            self.conv3_1x1 = nn.Conv2d(
                in_channels=self.inter_channels,
                out_channels=self.inter_channels * self.expansion,
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.batchnorm3 = nn.BatchNorm2d(self.inter_channels * self.expansion)

        else:
            # Basic block
            # 3x3 conv
            self.conv1_3x3 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.batchnorm1 = nn.BatchNorm2d(self.inter_channels)

            # 3x3 conv
            self.conv2_3x3 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )
            self.batchnorm2 = nn.BatchNorm2d(self.inter_channels)

    def forward(self, x):
        # Input stored to be added before the final ReLU
        in_x = x

        if self.is_bottleneck:
            # conv1x1 -> BN -> ReLU
            x = self.relu(self.batchnorm1(self.conv1_1x1(x)))

            # conv3x3 -> BN -> ReLU
            x = self.relu(self.batchnorm2(self.conv2_3x3(x)))

            # conv1x1 -> BN
            x = self.batchnorm3(self.conv3_1x1(x))

        else:
            # conv3x3 -> BN -> ReLU
            x = self.relu(self.batchnorm1(self.conv1_3x3(x)))

            # conv3x3 -> BN
            x = self.batchnorm2(self.conv2_3x3(x))

        if self.identity:
            x += in_x
        else:
            x += self.projection(in_x)

        # Final ReLU
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    """
    Create ResNet architecture based on the provided variant (18/34/50/101/etc)
    """

    def __init__(self, resnet_variant, in_channels, num_classes):
        """

        Args:
            resnet_variant: list -
            in_channels: int - image channels (3)
            num_classes: int - number of classes
        """
        super().__init__()
        self.channels_list = resnet_variant[0]
        self.repetition_list = resnet_variant[1]
        self.expansion = resnet_variant[2]
        self.is_bottleneck = resnet_variant[3]

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self._make_blocks(
            in_channels=64,
            inter_channels=self.channels_list[0],
            num_repeat=self.repetition_list[0], expansion=self.expansion, is_bottleneck=self.is_bottleneck, stride=1
        )
        self.block2 = self._make_blocks(
            in_channels=self.channels_list[0] * self.expansion,
            inter_channels=self.channels_list[1],
            num_repeat=self.repetition_list[1], expansion=self.expansion, is_bottleneck=self.is_bottleneck, stride=2
        )
        self.block3 = self._make_blocks(
            in_channels=self.channels_list[1] * self.expansion,
            inter_channels=self.channels_list[2],
            num_repeat=self.repetition_list[2], expansion=self.expansion, is_bottleneck=self.is_bottleneck, stride=2
        )
        self.block4 = self._make_blocks(
            in_channels=self.channels_list[2] * self.expansion,
            inter_channels=self.channels_list[3],
            num_repeat=self.repetition_list[3], expansion=self.expansion, is_bottleneck=self.is_bottleneck, stride=2
        )

        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.channels_list[3] * self.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.average_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x

    def _make_blocks(self, in_channels, inter_channels, num_repeat, expansion, is_bottleneck, stride):
        """

        Args:
            in_channels: int - number of channels of the bottleneck input
            inter_channels: int - number of channels of the 3x3 conv in the bottleneck
            num_repeat: int - number of bottlenecks in the block
            expansion: int - factor by which the intermediate channels are multiplied to create the output channels
            is_bottleneck: boolean - status if bottleneck is required
            stride: int - stride to be used in the first bottleneck 3x3 conv

        Returns:

        """
        layers = []
        layers.append(Bottleneck(in_channels, inter_channels, expansion, is_bottleneck, stride))
        for num in range(1, num_repeat):
            layers.append(Bottleneck(inter_channels * expansion, inter_channels, expansion, is_bottleneck, stride=1))

        return nn.Sequential(*layers)
