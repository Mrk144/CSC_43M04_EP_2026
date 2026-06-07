from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from models.temporal_utils import temporal_interpolate

class VideoMAEClassifier(nn.Module):

    def __init__(self, variant: str='MCG-NJU/videomae-base-finetuned-ssv2', num_classes: int=33, pretrained: bool=True, freeze_backbone: bool=False, ignore_mismatched_sizes: bool=True, num_frames: int=16) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        try:
            from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        except ImportError as e:
            raise ImportError("VideoMAEClassifier requires the 'transformers' package. Install it via `uv add transformers` or `pip install transformers`.") from e
        self.variant = variant
        self.num_classes = num_classes
        if pretrained:
            self.model = VideoMAEForVideoClassification.from_pretrained(variant, num_labels=num_classes, ignore_mismatched_sizes=ignore_mismatched_sizes)
        else:
            cfg = VideoMAEConfig.from_pretrained(variant, num_labels=num_classes)
            self.model = VideoMAEForVideoClassification(cfg)
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def classifier(self) -> nn.Module:
        return self.model.classifier

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    @property
    def expected_num_frames(self) -> int:
        return int(getattr(self.model.config, 'num_frames', 16))

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.dim() != 5:
            raise ValueError(f'Expected video_batch with 5 dims (B,T,C,H,W), got {video_batch.shape}')
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        outputs = self.model(pixel_values=video_batch)
        return outputs.logits

    def load_state_dict(self, state_dict: dict, strict: bool=True, assign: bool=False) -> Optional[object]:
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
