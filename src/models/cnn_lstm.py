"""
CNN + BiLSTM with attention pooling.

ResNet18 produces a per-frame feature vector; a bidirectional LSTM reads the
short sequence (T frames); a learnable attention pool aggregates the sequence
into a single vector that is fed to the classifier. With T=4 frames, attention
pooling is markedly better than reading only the last LSTM hidden state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class AttentionPool1d(nn.Module):
    """Soft attention over the temporal dim of a (B, T, D) sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)
        nn.init.normal_(self.score.weight, 0, 0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x)
        weights = torch.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        lstm_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = 2 * lstm_hidden_size
        self.attn_pool = AttentionPool1d(pooled_dim)
        self.classifier = nn.Linear(pooled_dim, num_classes)

        self._init_lstm_and_head(pretrained=pretrained)

    def _init_lstm_and_head(self, pretrained: bool) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)
        if not pretrained:
            for m in self.backbone.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video_batch.shape
        frames = video_batch.reshape(b * t, c, h, w)
        frame_features = self.backbone(frames)
        frame_features = torch.flatten(frame_features, start_dim=1)
        sequence = frame_features.view(b, t, -1)
        lstm_out, _ = self.lstm(sequence)
        pooled = self.attn_pool(lstm_out)
        return self.classifier(pooled)
