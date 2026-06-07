from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from models.temporal_utils import temporal_interpolate

def _trunc_normal_(tensor: torch.Tensor, std: float=0.02) -> None:
    init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

class DropPath(nn.Module):

    def __init__(self, drop_prob: float=0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        hidden = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.Dropout(dropout))
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        (attn_out, _) = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class CNNTransformer(nn.Module):

    def __init__(self, num_classes: int, pretrained: bool=False, num_frames: int=7, spatial_tokens_side: int=1, d_model: int=512, num_layers: int=4, num_heads: int=8, mlp_ratio: float=2.0, dropout: float=0.1, attn_dropout: float=0.0, drop_path: float=0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by num_heads ({num_heads})')
        self.num_frames = int(num_frames)
        self.spatial_tokens_side = int(spatial_tokens_side)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        if feat_dim != d_model:
            raise ValueError(f'ResNet-18 produces {feat_dim}-d features but d_model={d_model}. Either keep d_model=512 or swap the backbone.')
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if self.spatial_tokens_side > 0:
            self.spatial_pool: Optional[nn.AdaptiveAvgPool2d] = nn.AdaptiveAvgPool2d(self.spatial_tokens_side)
            self.spatial_tokens = self.spatial_tokens_side ** 2
        else:
            self.spatial_pool = None
            self.spatial_tokens = 49
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max(self.num_frames, 1), 1, d_model))
        self.spatial_pos: Optional[nn.Parameter] = None
        if self.spatial_tokens > 1:
            self.spatial_pos = nn.Parameter(torch.zeros(1, 1, self.spatial_tokens, d_model))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_dropout = nn.Dropout(dropout)
        drop_path_rates = [float(x) for x in torch.linspace(0.0, drop_path, steps=max(num_layers, 1))]
        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, attn_dropout=attn_dropout, drop_path=drop_path_rates[i]) for i in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))
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
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    init.ones_(m.weight)
                    init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        names = {'cls_token', 'temporal_pos', 'cls_pos'}
        if self.spatial_pos is not None:
            names.add('spatial_pos')
        return names

    def _extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(frames)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

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
        cls_out = tokens[:, 0]
        return self.classifier(cls_out)
__all__ = ['CNNTransformer']
