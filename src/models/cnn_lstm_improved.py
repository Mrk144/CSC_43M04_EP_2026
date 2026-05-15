"""
CNNLSTMImproved: same backbone+BiLSTM+attention as ``CNNLSTM`` plus a GELU
activation and dropout between the backbone and the LSTM, and a dropout in the
classification head. Fixes a previous bug where ``self.apply(...)`` re-ran
Kaiming init over the entire backbone, including pretrained ResNet weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

from models.cnn_lstm import AttentionPool1d
from models.temporal_utils import temporal_interpolate


class CNNLSTMImproved(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        lstm_hidden_size: int = 256,
        dropout_p: float = 0.5,
        num_frames: int = 0,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.feature_dropout = nn.Dropout(p=dropout_p)
        self.activation = nn.GELU()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        pooled_dim = 2 * lstm_hidden_size
        self.attn_pool = AttentionPool1d(pooled_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(pooled_dim, num_classes),
        )

        self._init_added_modules(pretrained=pretrained)

    def _init_added_modules(self, pretrained: bool) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                init.orthogonal_(param.data)
            elif "bias" in name:
                init.constant_(param.data, 0)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        if not pretrained:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        b, t, c, h, w = video_batch.shape
        frames = video_batch.reshape(b * t, c, h, w)

        frame_features = self.backbone(frames)
        frame_features = self.activation(frame_features)
        frame_features = self.feature_dropout(frame_features)

        sequence = frame_features.view(b, t, -1)
        lstm_out, _ = self.lstm(sequence)
        pooled = self.attn_pool(lstm_out)
        return self.classifier(pooled)
