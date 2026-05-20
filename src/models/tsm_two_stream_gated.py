from __future__ import annotations

import torch
import torch.nn as nn

from models.tsm_resnet import TSMResNet


class TSMTwoStreamGated(nn.Module):
    """Two-stream TSM with learned per-feature fusion gate."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 7,
        pretrained: bool = False,
        dropout_p: float = 0.5,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
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
        self.rgb_stream.fc = nn.Identity()
        self.diff_stream.fc = nn.Identity()

        dim = 512
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(dim, num_classes)
        nn.init.normal_(self.classifier.weight, 0.0, 0.01)
        nn.init.constant_(self.classifier.bias, 0.0)

    @staticmethod
    def _frame_diff(video_batch: torch.Tensor) -> torch.Tensor:
        diff = video_batch[:, 1:] - video_batch[:, :-1]
        pad = torch.zeros_like(video_batch[:, :1])
        return torch.cat([pad, diff], dim=1)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        diff = self._frame_diff(video_batch)
        rgb_feat = self.rgb_stream.encode(video_batch)
        diff_feat = self.diff_stream.encode(diff)
        g = self.gate(torch.cat([rgb_feat, diff_feat], dim=1))
        fused = g * rgb_feat + (1.0 - g) * diff_feat
        return self.classifier(self.dropout(fused))
