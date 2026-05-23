"""
EfficientFormer V2 (timm) per-frame encoder + BiLSTM + attention pooling (Track A).

Expects ImageNet-normalized inputs via ``build_transforms``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from models.cnn_lstm_improved import AttentionPool1d
from models.temporal_utils import temporal_interpolate

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "efficientformer_bilstm requires the 'timm' package. Install with: uv add timm"
    ) from e


class EfficientFormerBiLSTM(nn.Module):
    """Per-frame EfficientFormer V2 + bidirectional LSTM + temporal attention."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        num_frames: int = 7,
        variant: str = "efficientformerv2_s1",
        lstm_hidden_size: int = 256,
        dropout_p: float = 0.5,
        backbone_chunk_size: int = 28,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.backbone_chunk_size = max(1, int(backbone_chunk_size))
        self.variant = variant

        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = int(getattr(self.backbone, "num_features", 0))
        if feat_dim <= 0:
            raise RuntimeError(
                f"Could not read num_features from timm model {variant!r}"
            )
        self.feature_dim = feat_dim

        self.feature_dropout = nn.Dropout(p=dropout_p)
        self.activation = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
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
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    if hasattr(m, "weight") and m.weight is not None:
                        init.constant_(m.weight, 1)
                    if getattr(m, "bias", None) is not None:
                        init.constant_(m.bias, 0)

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames``: (B*T, C, H, W) -> (B*T, feature_dim), chunked for VRAM."""
        chunk = self.backbone_chunk_size
        n = frames.size(0)
        if n <= chunk:
            return self.backbone(frames)
        return torch.cat(
            [self.backbone(frames[i : i + chunk]) for i in range(0, n, chunk)],
            dim=0,
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        b, t, c, h, w = video_batch.shape
        frames = video_batch.reshape(b * t, c, h, w)
        frame_features = self._encode_frames(frames)
        frame_features = self.activation(frame_features)
        frame_features = self.feature_dropout(frame_features)
        sequence = frame_features.view(b, t, -1)
        lstm_out, _ = self.lstm(sequence)
        pooled = self.attn_pool(lstm_out)
        return self.classifier(pooled)
