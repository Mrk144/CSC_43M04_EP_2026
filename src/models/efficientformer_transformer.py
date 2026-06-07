from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.cnn_transformer import TransformerBlock, _trunc_normal_
from models.temporal_utils import temporal_interpolate
try:
    import timm
except ImportError as e:
    raise ImportError("efficientformer_transformer requires the 'timm' package. Install with: uv add timm") from e

class EfficientFormerTransformer(nn.Module):

    def __init__(self, num_classes: int, pretrained: bool=False, num_frames: int=7, variant: str='efficientformerv2_s1', spatial_tokens_side: int=1, num_layers: int=4, num_heads: int=8, mlp_ratio: float=2.0, dropout: float=0.1, attn_dropout: float=0.0, drop_path: float=0.1, backbone_chunk_size: int=28) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.backbone_chunk_size = max(1, int(backbone_chunk_size))
        self.spatial_tokens_side = int(spatial_tokens_side)
        self.variant = variant
        self.backbone = timm.create_model(variant, pretrained=pretrained, num_classes=0, global_pool='')
        feat_dim = int(getattr(self.backbone, 'num_features', 0))
        if feat_dim <= 0:
            raise RuntimeError(f'Could not read num_features from timm model {variant!r}')
        self.d_model = feat_dim
        if self.d_model % num_heads != 0:
            raise ValueError(f'num_features ({self.d_model}) must be divisible by num_heads ({num_heads}) for {variant!r}')
        if self.spatial_tokens_side > 0:
            self.spatial_pool: Optional[nn.AdaptiveAvgPool2d] = nn.AdaptiveAvgPool2d(self.spatial_tokens_side)
            self.spatial_tokens = self.spatial_tokens_side ** 2
        else:
            self.spatial_pool = None
            self.spatial_tokens = 49
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max(self.num_frames, 1), 1, self.d_model))
        self.spatial_pos: Optional[nn.Parameter] = None
        if self.spatial_tokens > 1:
            self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.spatial_tokens, self.d_model))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.input_dropout = nn.Dropout(dropout)
        drop_path_rates = [float(x) for x in torch.linspace(0.0, drop_path, steps=max(num_layers, 1))]
        self.blocks = nn.ModuleList([TransformerBlock(d_model=self.d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout, drop_path=drop_path_rates[i]) for i in range(num_layers)])
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_model, num_classes))
        self._init_added_modules(pretrained=pretrained)

    def _init_added_modules(self, pretrained: bool) -> None:
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.temporal_pos, std=0.02)
        _trunc_normal_(self.cls_pos, std=0.02)
        if self.spatial_pos is not None:
            _trunc_normal_(self.spatial_pos, std=0.02)
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
        if not pretrained:
            for m in self.backbone.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        if isinstance(m, nn.Conv2d):
                            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        else:
                            _trunc_normal_(m.weight, std=0.02)
                    if getattr(m, 'bias', None) is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        init.ones_(m.weight)
                    if getattr(m, 'bias', None) is not None:
                        init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        names = {'cls_token', 'temporal_pos', 'cls_pos'}
        if self.spatial_pos is not None:
            names.add('spatial_pos')
        return names

    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        chunk = self.backbone_chunk_size
        n = frames.size(0)
        if n <= chunk:
            chunks = [self._backbone_forward_frames(frames)]
        else:
            chunks = [self._backbone_forward_frames(frames[i:i + chunk]) for i in range(0, n, chunk)]
        return torch.cat(chunks, dim=0)

    def _backbone_forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(frames)
        if feat.dim() == 2:
            side = int(round(feat.shape[1] ** 0.5))
            if side * side == feat.shape[1]:
                feat = feat.view(feat.shape[0], side, side, -1).permute(0, 3, 1, 2)
            else:
                feat = feat.unsqueeze(-1).unsqueeze(-1)
        return feat

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        if self.num_frames > 0:
            video_batch = temporal_interpolate(video_batch, self.num_frames)
        (b, t, c, h, w) = video_batch.shape
        frames = video_batch.reshape(b * t, c, h, w)
        feat = self._extract_frame_features(frames)
        if self.spatial_pool is not None:
            feat = self.spatial_pool(feat)
        (bt, feat_c, fh, fw) = feat.shape
        s = fh * fw
        tokens = feat.permute(0, 2, 3, 1).reshape(b, t, s, feat_c)
        temp_pos = self.temporal_pos
        if temp_pos.shape[1] != t:
            temp_pos = F.interpolate(temp_pos.permute(0, 3, 1, 2), size=(t, temp_pos.shape[2]), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        tokens = tokens + temp_pos
        if self.spatial_pos is not None:
            spatial_pos = self.spatial_pos
            if spatial_pos.shape[2] != s:
                side = int(round(spatial_pos.shape[2] ** 0.5))
                target_side = int(round(s ** 0.5))
                if side * side == spatial_pos.shape[2] and target_side * target_side == s:
                    grid = spatial_pos.permute(0, 3, 1, 2).reshape(1, feat_c, 1, side, side)[:, :, 0]
                    grid = F.interpolate(grid, size=(target_side, target_side), mode='bilinear', align_corners=False)
                    spatial_pos = grid.reshape(1, feat_c, target_side * target_side).permute(0, 2, 1).unsqueeze(0)
                else:
                    spatial_pos = None
            if spatial_pos is not None:
                tokens = tokens + spatial_pos
        tokens = tokens.reshape(b, t * s, feat_c)
        cls = self.cls_token.expand(b, -1, -1) + self.cls_pos
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return self.classifier(tokens[:, 0])
