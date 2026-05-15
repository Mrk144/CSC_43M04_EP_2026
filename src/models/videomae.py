"""VideoMAE wrapper for Track B (open world).

Loads a Kinetics/SSv2-pretrained VideoMAE ViT from HuggingFace and exposes a
plug-and-play classifier compatible with the rest of the codebase
(``(B, T, C, H, W) -> (B, num_classes)``).

The base VideoMAE expects 16 frames at 224x224. The dataset serves the raw
``T = 4`` frames from disk; we upsample to ``self.num_frames`` (16 by default)
inside the wrapper. This is critical for submission: only the ``.pt`` is
shipped, so the temporal transformation has to be inside the model graph.

Variants
--------
- ``MCG-NJU/videomae-base-finetuned-ssv2`` (default; closest to the task)
- ``MCG-NJU/videomae-base-finetuned-kinetics``
- ``MCG-NJU/videomae-large-finetuned-kinetics``
- ``MCG-NJU/videomae-base-finetuned-ssv2-fps2`` (alternative SSv2 finetune)

Notes
-----
- Normalization: VideoMAE uses ImageNet mean/std, which matches our
  ``build_transforms(use_imagenet_norm=True)`` path activated when
  ``model.pretrained=true``.
- For phase 1 (linear probing), pass ``freeze_backbone=True``: only the
  classification head is trained.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.temporal_utils import temporal_interpolate


class VideoMAEClassifier(nn.Module):
    def __init__(
        self,
        variant: str = "MCG-NJU/videomae-base-finetuned-ssv2",
        num_classes: int = 33,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        ignore_mismatched_sizes: bool = True,
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        try:
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        except ImportError as e:
            raise ImportError(
                "VideoMAEClassifier requires the 'transformers' package. "
                "Install it via `uv add transformers` or `pip install transformers`."
            ) from e

        self.variant = variant
        self.num_classes = num_classes

        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                variant,
                num_labels=num_classes,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
        else:
            cfg = VideoMAEConfig.from_pretrained(variant, num_labels=num_classes)
            self.model = VideoMAEForVideoClassification(cfg)

        if freeze_backbone:
            self.freeze_backbone()

    @property
    def classifier(self) -> nn.Module:
        """Expose the classification head without registering it twice.

        Used to be ``self.classifier = self.model.classifier`` but that
        registered the same submodule under two names and produced duplicate
        keys in ``state_dict()``. A property avoids the registration while
        keeping the public API.
        """
        return self.model.classifier

    def freeze_backbone(self) -> None:
        """Freeze every parameter except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    @property
    def expected_num_frames(self) -> int:
        # VideoMAE uses temporal_patch_size=2 and num_frames=16 by default.
        return int(getattr(self.model.config, "num_frames", 16))

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """``video_batch``: ``(B, T, C, H, W)`` -> logits ``(B, num_classes)``.

        Input ``T`` can differ from VideoMAE's expected number of frames
        (typically 16): we linearly interpolate to ``self.num_frames`` first.
        HuggingFace's VideoMAE expects ``pixel_values`` of shape
        ``(B, T, C, H, W)`` (channel-second after time).
        """
        if video_batch.dim() != 5:
            raise ValueError(
                f"Expected video_batch with 5 dims (B,T,C,H,W), got {video_batch.shape}"
            )

        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)

        outputs = self.model(pixel_values=video_batch)
        return outputs.logits

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
        assign: bool = False,
    ) -> Optional[object]:
        """Forward to nn.Module's loader so checkpoints with our top-level keys work."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
