"""
Compact video transformer for Track A (from scratch, ~2–4M params).

Lighter than ``CNNTransformer`` (no ResNet-18): a small conv stem extracts one
token per frame, then a short temporal transformer with CLS classifies the clip.

Pipeline:
    (B, T_raw, 3, H, W) -> temporal_interpolate -> (B, T, 3, H, W)
    -> shared conv stem per frame -> (B, T, d_model)
    -> + temporal position embedding
    -> prepend CLS -> L x TransformerBlock -> Linear(num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.cnn_transformer import TransformerBlock
from models.temporal_utils import temporal_interpolate


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class ConvStem(nn.Module):
    """Lightweight 3-stage conv trunk -> one vector per frame."""

    def __init__(self, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames``: (B*T, 3, H, W) -> (B*T, out_dim)."""
        return self.net(frames)


class CompactVideoTransformer(nn.Module):
    """
    Compact temporal transformer over per-frame tokens.

    Args:
        num_classes: output classes (33 for this challenge).
        pretrained: ignored (stem is always trained from scratch); kept for API
            parity with other models.
        num_frames: internal T after interpolation (default 7). ``0`` = no interp.
        d_model: token width (default 256).
        num_layers: transformer depth (default 3).
        num_heads: attention heads (default 4).
        mlp_ratio: FFN expansion (default 2.0).
        dropout: dropout on stem, transformer MLP, classifier.
        attn_dropout: attention dropout.
        drop_path: max stochastic depth rate (linearly scaled across blocks).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        num_frames: int = 7,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.15,
        attn_dropout: float = 0.0,
        drop_path: float = 0.05,
    ) -> None:
        super().__init__()
        del pretrained  # no external backbone; train end-to-end from scratch

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.num_frames = int(num_frames)
        self.d_model = int(d_model)

        self.stem = ConvStem(out_dim=d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, max(self.num_frames, 1), d_model)
        )
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_dropout = nn.Dropout(dropout)

        rates = [
            float(x)
            for x in torch.linspace(0.0, drop_path, steps=max(num_layers, 1))
        ]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    drop_path=rates[i],
                )
                for i in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.temporal_pos, std=0.02)
        _trunc_normal_(self.cls_pos, std=0.02)

        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        for block in self.blocks:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    _trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    init.ones_(m.weight)
                    init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"cls_token", "temporal_pos", "cls_pos"}

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_batch: (B, T_raw, 3, H, W).
        Returns:
            logits (B, num_classes).
        """
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)

        b, t, c, h, w = video_batch.shape
        frames = video_batch.reshape(b * t, c, h, w)
        tokens = self.stem(frames).reshape(b, t, self.d_model)

        pos = self.temporal_pos
        if pos.shape[1] != t:
            pos = F.interpolate(
                pos.transpose(1, 2),
                size=t,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        tokens = tokens + pos

        cls = self.cls_token.expand(b, -1, -1) + self.cls_pos
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.input_dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.classifier(tokens[:, 0])


__all__ = ["CompactVideoTransformer"]


def _sanity_check() -> None:  # pragma: no cover
    torch.manual_seed(0)
    m = CompactVideoTransformer(num_classes=33, num_frames=7)
    x = torch.randn(2, 4, 3, 112, 112)
    y = m(x)
    assert y.shape == (2, 33), y.shape
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"ok: out={tuple(y.shape)}, trainable params={n / 1e6:.2f}M")


if __name__ == "__main__":  # pragma: no cover
    _sanity_check()
