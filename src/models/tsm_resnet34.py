from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.temporal_utils import temporal_interpolate
from models.tsm_resnet import install_temporal_shift


class TSMResNet34(nn.Module):
    """TSM over a ResNet34 backbone (from scratch friendly)."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 7,
        pretrained: bool = False,
        dropout_p: float = 0.5,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)
        install_temporal_shift(resnet, num_frames=num_frames, fold_div=fold_div)

        feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.pool_dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(feature_dim, num_classes)

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
        feats = self.backbone(video_batch.reshape(b * t, c, h, w)).view(b, t, -1).mean(dim=1)
        return feats

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        feats = self.pool_dropout(self.encode(video_batch))
        return self.fc(feats)
