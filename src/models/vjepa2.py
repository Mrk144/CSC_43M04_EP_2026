from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_utils import temporal_interpolate

def _is_ssv2_classification_variant(variant: str) -> bool:
    v = variant.lower()
    return 'ssv2' in v or v.endswith('-ssv2')

class VJEPA2Classifier(nn.Module):

    def __init__(self, variant: str='facebook/vjepa2-vitl-fpc16-256-ssv2', num_classes: int=33, pretrained: bool=True, freeze_backbone: bool=False, ignore_mismatched_sizes: bool=True, num_frames: int=16, input_size: int=256, dropout_p: float=0.1) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForVideoClassification
        except ImportError as e:
            raise ImportError("VJEPA2Classifier requires the 'transformers' package (>=4.52 for V-JEPA2 support). Install with `uv add 'transformers>=4.52'`.") from e
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames)
        self.input_size = int(input_size)
        self.dropout_p = float(dropout_p)
        self._use_hf_classifier = _is_ssv2_classification_variant(variant)
        if self._use_hf_classifier:
            if pretrained:
                self.model = AutoModelForVideoClassification.from_pretrained(variant, num_labels=num_classes, ignore_mismatched_sizes=ignore_mismatched_sizes)
            else:
                config = AutoConfig.from_pretrained(variant, num_labels=num_classes)
                self.model = AutoModelForVideoClassification.from_config(config)
            self.backbone = None
            self.classifier = None
        else:
            if pretrained:
                self.backbone = AutoModel.from_pretrained(variant)
            else:
                config = AutoConfig.from_pretrained(variant)
                self.backbone = AutoModel.from_config(config)
            self.model = None
            hidden_size: Optional[int] = None
            for attr in ('hidden_size', 'embed_dim'):
                if hasattr(self.backbone.config, attr):
                    hidden_size = int(getattr(self.backbone.config, attr))
                    break
            if hidden_size is None:
                raise ValueError(f'Could not infer hidden size from V-JEPA2 config for {variant}')
            self.hidden_size = hidden_size
            self.dropout = nn.Dropout(p=dropout_p)
            self.classifier = nn.Linear(hidden_size, num_classes)
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            nn.init.constant_(self.classifier.bias, 0)
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        if self._use_hf_classifier:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.classifier.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        if self._use_hf_classifier:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def _spatial_resize(self, video: torch.Tensor) -> torch.Tensor:
        (b, t, c, h, w) = video.shape
        if h == self.input_size and w == self.input_size:
            return video
        flat = video.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return flat.reshape(b, t, c, self.input_size, self.input_size)

    def _pool_tokens(self, last_hidden: torch.Tensor) -> torch.Tensor:
        if last_hidden.dim() == 2:
            return last_hidden
        if last_hidden.dim() == 3:
            return last_hidden.mean(dim=1)
        raise ValueError(f'Unexpected V-JEPA2 hidden state shape: {tuple(last_hidden.shape)}')

    def _forward_encoder(self, video_batch: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values_videos=video_batch)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            features = outputs[0]
        pooled = self._pool_tokens(features)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def _forward_classification(self, video_batch: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.model(pixel_values_videos=video_batch)
        except TypeError:
            outputs = self.model(pixel_values=video_batch)
        return outputs.logits

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.dim() != 5:
            raise ValueError(f'Expected (B, T, C, H, W) input, got {tuple(video_batch.shape)}')
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        video_batch = self._spatial_resize(video_batch)
        if self._use_hf_classifier:
            return self._forward_classification(video_batch)
        return self._forward_encoder(video_batch)
