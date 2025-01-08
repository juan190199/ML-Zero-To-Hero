import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x, rois):
        batch_size, channels, height, width = x.size()
        pooled_outputs = []

        if rois is not None:
            rois = rois.long()
            for roi in rois:
                batch_idx, x_min, y_min, x_max, y_max = roi

                # Ensure indices are within bounds
                x_min = torch.clamp(x_min, min=0)
                y_min = torch.clamp(y_min, min=0)
                x_max = torch.clamp(x_max, max=width)
                y_max = torch.clamp(y_max, max=height)

                roi_feature = x[batch_idx, :, y_min:y_max, x_min:x_max]

                roi_pooled = []
                for pool_size in self.pool_sizes:
                    pool = F.adaptive_max_pool2d(roi_feature, (pool_size, pool_size))
                    roi_pooled.append(torch.flatten(pool, 1))

                pooled_outputs.append(torch.cat(roi_pooled, dim=1))

        else:
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
        self.spatial_scale_h = None
        self.spatial_scale_w = None

    def forward(self, x, rois=None):
        """

        Args:
            x:
                input image batch
            rois:
                List of RoIs, where each RoI is of the form [batch_idx, x_min, y_min, x_max, y_max]

        Returns:

        """
        input_h, input_w = x.size(2), x.size(3)
        x = self.feature_extractor(x)
        if self.spatial_scale_h is None or self.spatial_scale_w is None:
            x_h, x_w = x.size(2), x.size(3)
            self.spatial_scale_h = x_h / input_h
            self.spatial_scale_w = x_w / input_w

        if rois is not None:
            rois = rois.clone()
            rois[:, 1] *= self.spatial_scale_w
            rois[:, 2] *= self.spatial_scale_h
            rois[:, 3] *= self.spatial_scale_w
            rois[:, 4] *= self.spatial_scale_h
            x = self.extract_rois(x, rois)

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

    def extract_rois(self, x, rois):
        pooled_features = []
        for roi in rois:
            batch_idx, x_min, y_min, x_max, y_max = roi

            x_min, y_min, x_max, y_max = torch.round(x_min), torch.round(y_min), torch.round(x_max), torch.round(y_max)
            x_min = max(0, x_min.item())
            y_min = max(0, y_min.item())
            x_max = min(x.size(3), x_max.item())
            y_max = min(x.size(2), y_max.item())
            batch_idx = int(batch_idx.item())

            roi_feature = x[batch_idx, :, y_min:y_max, x_min:x_max]
            pooled_features.append(roi_feature)

        return torch.cat(pooled_features, dim=0)
