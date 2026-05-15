"""PyTorchVideo X3D wrapper for Track B.

X3D (Feichtenhofer et al., 2020) is a family of efficient 3D-CNNs that
expand from a small 2D image network along multiple axes (depth, width,
temporal, etc.). Pretrained on Kinetics-400 with strong motion features and
much smaller than VideoMAE / V-JEPA2 ; great as an *ensemble member* that
brings a different inductive bias (convolutional, dense temporal) compared to
the ViT-based foundation models.

Variants (input ``T x H x W``)
------------------------------
- ``x3d_xs`` : 4   x 182 x 182  (smallest, perfect for very-short clips)
- ``x3d_s``  : 13  x 182 x 182
- ``x3d_m``  : 16  x 224 x 224  (default, sweet spot at 224)
- ``x3d_l``  : 16  x 312 x 312

Notes
-----
- Loaded via ``torch.hub.load('facebookresearch/pytorchvideo', name, ...)``,
  which clones the repo into the hub cache. The ``pytorchvideo`` package on
  PyPI provides the necessary Python modules at import time.
- X3D expects input shape ``(B, C, T, H, W)`` — we permute internally.
- The classifier head sits in ``model.blocks[-1].proj`` (a Linear from
  feature_dim to ``num_classes_kinetics=400``); we swap it for a fresh head.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_utils import temporal_interpolate


# Mapping variant -> (recommended num_frames, recommended input_size).
_X3D_VARIANTS: Dict[str, Tuple[int, int]] = {
    "x3d_xs": (4, 182),
    "x3d_s": (13, 182),
    "x3d_m": (16, 224),
    "x3d_l": (16, 312),
}


def available_x3d_variants() -> list[str]:
    return sorted(_X3D_VARIANTS.keys())


def _replace_x3d_head(model: nn.Module, num_classes: int) -> None:
    """Swap the Kinetics-400 logits Linear in the last block with a fresh one."""
    head_proj = getattr(model.blocks[-1], "proj", None)
    if isinstance(head_proj, nn.Linear):
        in_features = head_proj.in_features
        new_proj = nn.Linear(in_features, num_classes)
        nn.init.normal_(new_proj.weight, 0, 0.01)
        nn.init.constant_(new_proj.bias, 0)
        model.blocks[-1].proj = new_proj
        return

    # Fall back to scanning the last block for any Linear and replace the last
    # one found (covers edge cases where X3D adds extra Linear layers).
    last_linear: nn.Linear | None = None
    last_name = ""
    for name, m in model.blocks[-1].named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
            last_name = name
    if last_linear is None:
        raise ValueError("Could not locate X3D classification head Linear layer.")
    parent: nn.Module = model.blocks[-1]
    parts = last_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    new_proj = nn.Linear(last_linear.in_features, num_classes)
    nn.init.normal_(new_proj.weight, 0, 0.01)
    nn.init.constant_(new_proj.bias, 0)
    setattr(parent, parts[-1], new_proj)


class X3DClassifier(nn.Module):
    """PyTorchVideo X3D wrapped for ``(B, T, C, H, W)`` inputs."""

    def __init__(
        self,
        variant: str = "x3d_m",
        num_classes: int = 33,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_frames: int = 0,
        input_size: int = 0,
    ) -> None:
        super().__init__()
        if variant not in _X3D_VARIANTS:
            raise ValueError(
                f"Unknown X3D variant '{variant}'. Available: {available_x3d_variants()}"
            )

        default_T, default_S = _X3D_VARIANTS[variant]
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames) if num_frames > 0 else default_T
        self.input_size = int(input_size) if input_size > 0 else default_S

        try:
            self.net = torch.hub.load(
                "facebookresearch/pytorchvideo",
                variant,
                pretrained=pretrained,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load X3D variant '{variant}' from torch.hub. "
                "Make sure `pytorchvideo` is installed (`uv add pytorchvideo`) "
                "and the repo cache is reachable."
            ) from e

        _replace_x3d_head(self.net, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze every parameter, then re-enable the head."""
        for p in self.net.parameters():
            p.requires_grad = False
        head_proj = getattr(self.net.blocks[-1], "proj", None)
        if isinstance(head_proj, nn.Linear):
            for p in head_proj.parameters():
                p.requires_grad = True
        else:
            for m in self.net.blocks[-1].modules():
                if isinstance(m, nn.Linear):
                    for p in m.parameters():
                        p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = True

    def _spatial_resize(self, video: torch.Tensor) -> torch.Tensor:
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

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.dim() != 5:
            raise ValueError(
                f"Expected (B, T, C, H, W) input, got {tuple(video_batch.shape)}"
            )

        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        video_batch = self._spatial_resize(video_batch)

        x = video_batch.permute(0, 2, 1, 3, 4).contiguous()
        out = self.net(x)
        if out.dim() > 2:
            out = out.flatten(1)
        return out
