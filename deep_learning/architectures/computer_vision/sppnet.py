import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_outputs = []

        for pool_size in self.pool_sizes:
            output_size = (pool_size, pool_size)
            pool = F.adaptive_max_pool2d(x, output_size=output_size)
            pooled_outputs.append(torch.flatten(pool, 1))
            # pooled_outputs.append(torch.view(batch_size, -1))

        return torch.cat(pooled_outputs, dim=1)


class SPPNet(nn.Module):
    def __init__(self, num_classes, backbone=None, pool_sizes=(1, 2, 4)):
        super(SPPNet, self).__init__()
        self.num_classes = num_classes

        self.feature_extractor = backbone if backbone is not None else nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.spp = SpatialPyramidPooling(pool_sizes)

        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.spp(x)

        spp_output_size = x.size(1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(spp_output_size, 256)
            self.fc2 = nn.Linear(256, self.num_classes)

            # Register the layers with the module
            self.add_module('fc1', self.fc1)
            self.add_module('fc2', self.fc2)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
