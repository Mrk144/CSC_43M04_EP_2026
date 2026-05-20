from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.temporal_utils import temporal_interpolate
from models.tsm_resnet import install_temporal_shift


class TSMResNetRgbDiff(nn.Module):
    """Single-backbone TSM with RGB + motion-magnitude 4th channel."""

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

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self._patch_conv1_to_4ch(resnet, pretrained=pretrained)
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

    @staticmethod
    def _patch_conv1_to_4ch(resnet: nn.Module, pretrained: bool) -> None:
        old = resnet.conv1
        new = nn.Conv2d(
            4,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                new.weight[:, :3] = old.weight
                new.weight[:, 3:4] = old.weight.mean(dim=1, keepdim=True)
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        resnet.conv1 = new

    @staticmethod
    def _motion_channel(video_batch: torch.Tensor) -> torch.Tensor:
        diff = video_batch[:, 1:] - video_batch[:, :-1]
        mag = diff.abs().mean(dim=2, keepdim=True)
        pad = torch.zeros_like(mag[:, :1])
        return torch.cat([pad, mag], dim=1)

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
        motion = self._motion_channel(video_batch)
        x4 = torch.cat([video_batch, motion], dim=2)
        b, t, c, h, w = x4.shape
        feats = self.backbone(x4.reshape(b * t, c, h, w)).view(b, t, -1).mean(dim=1)
        return feats

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        feats = self.pool_dropout(self.encode(video_batch))
        return self.fc(feats)
