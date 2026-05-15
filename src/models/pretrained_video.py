"""Pretrained video backbones for Track B (open world).

Wraps ``torchvision.models.video`` backbones (Kinetics-400 pretrained) so they
fit the rest of the codebase: input ``(B, T, C, H, W)`` -> logits
``(B, num_classes)``.

To plug in a new pretrained model, add a builder function and register it in
``_REGISTRY`` below.

Caveats
-------
- Kinetics backbones expect input as ``(B, C, T, H, W)``: this wrapper permutes
  the input automatically.
- They were trained on long clips (T=16 or T=32). With this dataset's T=4 they
  still run mechanically. The 3D-conv variants (``r3d_18``, ``mc3_18``,
  ``r2plus1d_18``, ``s3d``) tolerate short clips most gracefully. Transformer
  variants (``mvit_*``, ``swin3d_*``) have positional embeddings tied to the
  original temporal extent and may fail or degrade with T != trained-T.
- The training code uses ImageNet normalization when ``model.pretrained=true``.
  Kinetics normalization is slightly different but ImageNet stats are a
  workable fallback (~0.5 pt accuracy difference in practice).
- All builders default to 224x224 inputs, which matches ``build_transforms``.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torchvision.models import video as tv_video


def _build_r3d_18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_mc3_18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.MC3_18_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.mc3_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_r2plus1d_18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_s3d(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.S3D_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.s3d(weights=weights)
    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv3d(
        in_channels, num_classes, kernel_size=(1, 1, 1)
    )
    return model


def _build_mvit_v1_b(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.mvit_v1_b(weights=weights)
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_mvit_v2_s(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.MViT_V2_S_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.mvit_v2_s(weights=weights)
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_swin3d_t(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.swin3d_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def _build_swin3d_s(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.Swin3D_S_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.swin3d_s(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def _build_swin3d_b(num_classes: int, pretrained: bool) -> nn.Module:
    weights = tv_video.Swin3D_B_Weights.KINETICS400_V1 if pretrained else None
    model = tv_video.swin3d_b(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


_REGISTRY: Dict[str, Callable[[int, bool], nn.Module]] = {
    # 3D convolutional (most robust to short clips)
    "r3d_18": _build_r3d_18,
    "mc3_18": _build_mc3_18,
    "r2plus1d_18": _build_r2plus1d_18,
    "s3d": _build_s3d,
    # Transformer (often best on K400 but sensitive to T)
    "mvit_v1_b": _build_mvit_v1_b,
    "mvit_v2_s": _build_mvit_v2_s,
    "swin3d_t": _build_swin3d_t,
    "swin3d_s": _build_swin3d_s,
    "swin3d_b": _build_swin3d_b,
}


def available_backbones() -> list[str]:
    """Return the list of supported pretrained backbones."""
    return sorted(_REGISTRY.keys())


class PretrainedVideoModel(nn.Module):
    """Wrap a torchvision video backbone for ``(B, T, C, H, W)`` inputs.

    Args:
        backbone: One of :func:`available_backbones`.
        num_classes: Number of output classes (head replacement).
        pretrained: Load Kinetics-400 weights if True (Track B only).
        freeze_backbone: If True, only the final classifier is trained
            (linear probing). Useful when data is small or the pretrained
            features are already strong.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if backbone not in _REGISTRY:
            raise ValueError(
                f"Unknown pretrained backbone '{backbone}'. "
                f"Available: {available_backbones()}"
            )
        self.backbone_name = backbone
        self.net = _REGISTRY[backbone](num_classes, pretrained)

        if freeze_backbone:
            self._freeze_all_but_head()

    def _freeze_all_but_head(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False

        if hasattr(self.net, "fc") and isinstance(self.net.fc, nn.Linear):
            for p in self.net.fc.parameters():
                p.requires_grad = True
        elif hasattr(self.net, "head"):
            head = self.net.head
            if isinstance(head, nn.Linear):
                for p in head.parameters():
                    p.requires_grad = True
            else:
                for module in head.modules():
                    if isinstance(module, nn.Linear):
                        for p in module.parameters():
                            p.requires_grad = True
        elif hasattr(self.net, "classifier"):
            for p in self.net.classifier.parameters():
                p.requires_grad = True

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        x = video_batch.permute(0, 2, 1, 3, 4).contiguous()
        out = self.net(x)
        if out.dim() > 2:
            out = out.flatten(1)
        return out
