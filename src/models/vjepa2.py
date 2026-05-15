"""V-JEPA2 wrapper for Track B.

V-JEPA2 (Meta, 2024) is a self-supervised video encoder trained on a very
large corpus of unlabeled video. It learns to predict the latent embeddings
of masked spatio-temporal regions and is one of the strongest backbones for
motion-heavy / fine-grained tasks such as Something-Something v2 (the closest
public benchmark to this challenge).

This module loads a V-JEPA2 encoder from HuggingFace, runs the (B, T, C, H, W)
clip through it, mean-pools the patch tokens, and attaches a fresh linear
classifier. Frame and spatial size adaptation is done inside ``forward`` so
that the resulting ``.pt`` is fully self-contained.

Variants
--------
- ``facebook/vjepa2-vitl-fpc16-256`` (default, ViT-L, 16 frames, 256x256)
- ``facebook/vjepa2-vitg-fpc16-256`` (ViT-g, heavier)
- ``facebook/vjepa2-vitl-fpc64-256`` (ViT-L, 64 frames, more temporal capacity)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_utils import temporal_interpolate


class VJEPA2Classifier(nn.Module):
    """Linear classifier on top of a frozen/finetuned V-JEPA2 encoder."""

    def __init__(
        self,
        variant: str = "facebook/vjepa2-vitl-fpc16-256",
        num_classes: int = 33,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_frames: int = 16,
        input_size: int = 256,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError(
                "VJEPA2Classifier requires the 'transformers' package "
                "(>=4.52 for V-JEPA2 support). "
                "Install with `uv add 'transformers>=4.52'`."
            ) from e

        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames)
        self.input_size = int(input_size)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(variant)
        else:
            config = AutoConfig.from_pretrained(variant)
            self.backbone = AutoModel.from_config(config)

        # Hidden size: V-JEPA2 ViT-L = 1024, ViT-g = 1408 (read from config).
        hidden_size: Optional[int] = None
        for attr in ("hidden_size", "embed_dim"):
            if hasattr(self.backbone.config, attr):
                hidden_size = int(getattr(self.backbone.config, attr))
                break
        if hidden_size is None:
            raise ValueError(
                f"Could not infer hidden size from V-JEPA2 config for {variant}"
            )
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _spatial_resize(self, video: torch.Tensor) -> torch.Tensor:
        """If H/W differ from ``self.input_size``, resize bilinearly per-frame."""
        b, t, c, h, w = video.shape
        if h == self.input_size and w == self.input_size:
            return video
        flat = video.reshape(b * t, c, h, w)
        flat = F.interpolate(
            flat,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return flat.reshape(b, t, c, self.input_size, self.input_size)

    def _pool_tokens(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """Mean-pool patch tokens.

        V-JEPA2 returns a sequence of patch tokens without a CLS token, so we
        average across all positions. Some HF heads return ``(B, N, D)``;
        others (rare) may already pool to ``(B, D)``.
        """
        if last_hidden.dim() == 2:
            return last_hidden
        if last_hidden.dim() == 3:
            return last_hidden.mean(dim=1)
        raise ValueError(
            f"Unexpected V-JEPA2 hidden state shape: {tuple(last_hidden.shape)}"
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.dim() != 5:
            raise ValueError(
                f"Expected (B, T, C, H, W) input, got {tuple(video_batch.shape)}"
            )

        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        video_batch = self._spatial_resize(video_batch)

        outputs = self.backbone(pixel_values_videos=video_batch)
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state
        else:
            features = outputs[0]

        pooled = self._pool_tokens(features)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
