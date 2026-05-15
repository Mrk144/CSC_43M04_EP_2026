"""
Temporal Shift Module on top of a ResNet18.

This implementation follows the original TSM paper:
- The shift is applied **inside the residual branch** of each BasicBlock (on the
  input of ``conv1``), never on the identity skip-connection. The skip path
  preserves the un-shifted features, which is what makes TSM ``residual``.
- For each BasicBlock, only a small fraction (``1/fold_div``) of the channels
  is shifted forward (toward the past) and another fraction backward (toward
  the future). The rest of the channels is left untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from models.temporal_utils import temporal_interpolate


def temporal_shift(x: torch.Tensor, num_frames: int, fold_div: int = 8) -> torch.Tensor:
    """Shift a fraction of channels along the temporal axis.

    Args:
        x: Tensor of shape ``(B * T, C, H, W)``.
        num_frames: Temporal extent ``T`` (must divide ``x.size(0)``).
        fold_div: Fraction ``1/fold_div`` of the channels to shift in each
            direction (forward + backward).
    """
    nt, c, h, w = x.size()
    n = nt // num_frames
    x = x.view(n, num_frames, c, h, w)
    fold = c // fold_div

    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(nt, c, h, w)


class TemporalShiftWrapper(nn.Module):
    """Apply a temporal shift to the input, then run the wrapped module.

    Designed to wrap ``BasicBlock.conv1`` so that the shift only affects the
    residual branch, not the skip-connection.
    """

    def __init__(self, net: nn.Module, num_frames: int, fold_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.num_frames = num_frames
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = temporal_shift(x, self.num_frames, self.fold_div)
        return self.net(x)


def install_temporal_shift(resnet: nn.Module, num_frames: int, fold_div: int = 8) -> None:
    """In-place: wrap every BasicBlock's ``conv1`` with a TemporalShift."""
    for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4):
        for block in layer:
            block.conv1 = TemporalShiftWrapper(
                block.conv1, num_frames=num_frames, fold_div=fold_div
            )


class TSMResNet(nn.Module):
    """ResNet18 with TSM injected inside every residual block.

    Forward signature matches the rest of the codebase: ``(B, T, C, H, W) -> (B, num_classes)``.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 4,
        pretrained: bool = False,
        dropout_p: float = 0.5,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
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
        """Return temporally pooled features ``(B, feature_dim)``.

        Input ``T`` may differ from ``self.num_frames`` (e.g. raw 4 frames at
        inference time): we linearly interpolate along the temporal axis so
        the backbone always sees ``self.num_frames`` frames.
        """
        video_batch = temporal_interpolate(video_batch, self.num_frames)
        b, t, c, h, w = video_batch.shape
        x = video_batch.reshape(b * t, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, t, -1).mean(dim=1)
        return feats

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        feats = self.encode(video_batch)
        feats = self.pool_dropout(feats)
        return self.fc(feats)
