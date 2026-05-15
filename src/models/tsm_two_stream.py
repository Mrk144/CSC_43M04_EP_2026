"""
Two-stream TSM-ResNet18: one branch on RGB frames, one on frame differences.

With only 4 frames per clip, the inter-frame differences carry most of the
motion signal that distinguishes directional classes. Late fusion concatenates
both pooled feature vectors and feeds them to a single linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.tsm_resnet import TSMResNet


class TSMTwoStream(nn.Module):
    """Two TSM-ResNet18 backbones fused at the feature level."""

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

        # Each stream produces a (B, 512) pooled feature vector via ``encode``.
        self.rgb_stream = TSMResNet(
            num_classes=num_classes,
            num_frames=num_frames,
            pretrained=pretrained,
            dropout_p=0.0,
            fold_div=fold_div,
        )
        self.diff_stream = TSMResNet(
            num_classes=num_classes,
            num_frames=num_frames,
            pretrained=pretrained,
            dropout_p=0.0,
            fold_div=fold_div,
        )

        feature_dim = 512  # ResNet18 output
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(feature_dim * 2, num_classes)

        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    @staticmethod
    def _frame_diff(video_batch: torch.Tensor) -> torch.Tensor:
        """Pad first slot with zeros so output keeps shape ``(B, T, C, H, W)``."""
        diff = video_batch[:, 1:] - video_batch[:, :-1]
        pad = torch.zeros_like(video_batch[:, :1])
        return torch.cat([pad, diff], dim=1)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        diff = self._frame_diff(video_batch)
        rgb_feats = self.rgb_stream.encode(video_batch)
        diff_feats = self.diff_stream.encode(diff)
        feats = torch.cat([rgb_feats, diff_feats], dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)
