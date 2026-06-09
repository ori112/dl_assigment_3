"""Convolutional network architectures for the pneumonia classifier.

We build two models:
  * ``TwoLayerConvNet`` - the required simple network (one conv layer + one
    hidden fully-connected layer), used as the convolutional baseline.
  * ``DeepConvNet``     - a deeper stack with max-pooling and batch
    normalization that we tune for the best performance.

Inputs are single-channel (grayscale) square images of side ``img_size``.
"""

import torch
import torch.nn as nn


class TwoLayerConvNet(nn.Module):
    """A simple conv-relu-maxpool feature stage followed by a 2-layer MLP head.

    Architecture:
        conv(3x3) -> ReLU -> maxpool(2) -> flatten -> fc -> ReLU -> fc
    """

    def __init__(self, in_channels=1, img_size=128, num_filters=16,
                 hidden_dim=100, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        # After one 2x2 pool the spatial size halves, so work out the flat size.
        pooled_size = img_size // 2
        flat_dim = num_filters * pooled_size * pooled_size

        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DeepConvNet(nn.Module):
    """A deeper network: several conv-batchnorm-relu blocks with max-pooling.

    Each entry in ``channels`` adds one block of:
        conv(3x3, pad 1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
    so the spatial resolution halves at every block while the channel count
    grows.  An adaptive average pool then fixes the spatial size before the
    classifier head, which keeps the head independent of the input resolution.

    Batch normalization stabilizes and speeds up training, and dropout in the
    head provides extra regularization - both help us tune for best performance.
    """

    def __init__(self, in_channels=1, channels=(32, 64, 128), num_classes=2,
                 head_pool=4, hidden_dim=256, dropout=0.5):
        super().__init__()

        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks.append(nn.Conv2d(prev_channels, out_channels,
                                    kernel_size=3, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d(kernel_size=2))
            prev_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Collapse to a fixed head_pool x head_pool grid regardless of img_size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((head_pool, head_pool))
        flat_dim = prev_channels * head_pool * head_pool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)
