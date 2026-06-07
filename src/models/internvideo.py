from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.temporal_utils import temporal_interpolate

def _try_extract_feature(out) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    for attr in ('pooler_output', 'last_hidden_state', 'image_embeds', 'video_embeds'):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            return getattr(out, attr)
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    raise ValueError(f'Could not extract features from InternVideo2 output: {type(out)}')

class InternVideo2Classifier(nn.Module):

    def __init__(self, variant: str='OpenGVLab/InternVideo2-Stage2_1B-224p-f8', num_classes: int=33, pretrained: bool=True, freeze_backbone: bool=False, num_frames: int=8, input_size: int=224, dropout_p: float=0.1) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError("InternVideo2Classifier requires the 'transformers' package. Install with `uv add 'transformers>=4.52'`.") from e
        self.variant = variant
        self.num_classes = int(num_classes)
        self.num_frames = int(num_frames)
        self.input_size = int(input_size)
        self._call_mode: Optional[str] = None
        if pretrained:
            self.backbone = AutoModel.from_pretrained(variant, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        hidden_size: Optional[int] = None
        for attr in ('hidden_size', 'embed_dim', 'projection_dim', 'vision_embed_dim'):
            if hasattr(self.backbone.config, attr):
                hidden_size = int(getattr(self.backbone.config, attr))
                break
        if hidden_size is None:
            with torch.no_grad():
                dummy = torch.zeros(1, num_frames, 3, input_size, input_size)
                out = self._call_backbone(dummy)
                feats = _try_extract_feature(out)
                hidden_size = feats.flatten(1).shape[1]
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

    def _call_backbone(self, video_batch: torch.Tensor):
        if self._call_mode == 'videos':
            return self.backbone(videos=video_batch)
        if self._call_mode == 'pixel_values':
            return self.backbone(pixel_values=video_batch)
        if self._call_mode == 'positional':
            return self.backbone(video_batch)
        last_error: Optional[BaseException] = None
        for (mode, fn) in (('videos', lambda : self.backbone(videos=video_batch)), ('pixel_values', lambda : self.backbone(pixel_values=video_batch)), ('positional', lambda : self.backbone(video_batch))):
            try:
                out = fn()
            except TypeError as exc:
                last_error = exc
                continue
            self._call_mode = mode
            return out
        raise RuntimeError(f'InternVideo2 backbone rejected videos=, pixel_values=, and positional inputs. Last TypeError: {last_error!r}')

    def _spatial_resize(self, video: torch.Tensor) -> torch.Tensor:
        (b, t, c, h, w) = video.shape
        if h == self.input_size and w == self.input_size:
            return video
        flat = video.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return flat.reshape(b, t, c, self.input_size, self.input_size)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.dim() != 5:
            raise ValueError(f'Expected (B, T, C, H, W) input, got {tuple(video_batch.shape)}')
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        video_batch = self._spatial_resize(video_batch)
        out = self._call_backbone(video_batch)
        feats = _try_extract_feature(out)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        elif feats.dim() > 2:
            feats = feats.flatten(1)
        feats = self.dropout(feats)
        return self.classifier(feats)
