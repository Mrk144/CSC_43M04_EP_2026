from __future__ import annotations

from models.tsm_backbone import TSMResNetBackbone


class TSMResNet50(TSMResNetBackbone):
    """TSM + ResNet50 (2048-d features)."""

    def __init__(
        self,
        num_classes: int,
        num_frames: int = 7,
        pretrained: bool = False,
        dropout_p: float = 0.5,
        fold_div: int = 8,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            backbone="resnet50",
            num_frames=num_frames,
            pretrained=pretrained,
            dropout_p=dropout_p,
            fold_div=fold_div,
        )
