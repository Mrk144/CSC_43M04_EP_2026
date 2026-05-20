from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.temporal_utils import temporal_interpolate
from models.tsm_resnet import install_temporal_shift


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg(x)
        w = self.fc2(self.act(self.fc1(w)))
        return x * self.gate(w)


class TSMResNetSE(nn.Module):
    """TSM-ResNet18 with SE channel attention after each stage."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 7,
        pretrained: bool = False,
        dropout_p: float = 0.5,
        fold_div: int = 8,
        se_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        install_temporal_shift(resnet, num_frames=num_frames, fold_div=fold_div)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.se1 = SEBlock(64, reduction=se_ratio)
        self.se2 = SEBlock(128, reduction=se_ratio)
        self.se3 = SEBlock(256, reduction=se_ratio)
        self.se4 = SEBlock(512, reduction=se_ratio)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.pool_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, num_classes)

        if not pretrained:
            self._initialize_weights()
        else:
            self._initialize_head_only()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_head_only(self) -> None:
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        video_batch = temporal_interpolate(video_batch, self.num_frames)
        b, t, c, h, w = video_batch.shape
        x = video_batch.reshape(b * t, c, h, w)
        x = self.stem(x)
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))
        x = self.global_pool(x).flatten(1)
        return x.view(b, t, -1).mean(dim=1)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        feats = self.pool_dropout(self.encode(video_batch))
        return self.fc(feats)
