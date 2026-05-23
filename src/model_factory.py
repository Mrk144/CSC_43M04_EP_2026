"""Instantiate models from Hydra ``cfg.model`` (shared by train and checkpoint loading)."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by ``cfg.model.name``.

    Imports are lazy so training can start without optional deps (e.g. ``timm``)
    until a model that needs them is selected.
    """
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    model_num_frames = int(cfg.model.get("num_frames", 0))

    if name == "cnn_lstm_improved":
        from models.cnn_lstm_improved import CNNLSTMImproved

        return CNNLSTMImproved(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 256)),
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            num_frames=model_num_frames,
        )
    if name == "cnn_transformer":
        from models.cnn_transformer import CNNTransformer

        ct_frames = model_num_frames if model_num_frames > 0 else 7
        return CNNTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=ct_frames,
            spatial_tokens_side=int(cfg.model.get("spatial_tokens_side", 1)),
            d_model=int(cfg.model.get("d_model", 512)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 2.0)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            drop_path=float(cfg.model.get("drop_path", 0.1)),
        )
    if name == "tsm_resnet":
        from models.tsm_resnet import TSMResNet

        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        return TSMResNet(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            fold_div=int(cfg.model.get("fold_div", 8)),
        )
    if name == "tsm_resnet34":
        from models.tsm_resnet34 import TSMResNet34

        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        return TSMResNet34(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            fold_div=int(cfg.model.get("fold_div", 8)),
        )
    if name == "tsm_resnet50":
        from models.tsm_resnet50 import TSMResNet50

        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        return TSMResNet50(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            fold_div=int(cfg.model.get("fold_div", 8)),
        )
    if name in ("tsm_two_stream_gated", "tsm_two_stream_gated_r50"):
        from models.tsm_two_stream_gated import TSMTwoStreamGated

        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        backbone = str(cfg.model.get("backbone", "resnet34"))
        if name == "tsm_two_stream_gated_r50":
            backbone = "resnet50"
        return TSMTwoStreamGated(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            fold_div=int(cfg.model.get("fold_div", 8)),
            backbone=backbone,
        )
    if name == "efficientformer_bilstm":
        from models.efficientformer_bilstm import EfficientFormerBiLSTM

        ef_frames = model_num_frames if model_num_frames > 0 else 7
        return EfficientFormerBiLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=ef_frames,
            variant=str(cfg.model.get("variant", "efficientformerv2_s1")),
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 256)),
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            backbone_chunk_size=int(cfg.model.get("backbone_chunk_size", 28)),
        )
    if name == "efficientformer_transformer":
        from models.efficientformer_transformer import EfficientFormerTransformer

        ef_frames = model_num_frames if model_num_frames > 0 else 7
        return EfficientFormerTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=ef_frames,
            variant=str(cfg.model.get("variant", "efficientformerv2_s1")),
            spatial_tokens_side=int(cfg.model.get("spatial_tokens_side", 1)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            mlp_ratio=float(cfg.model.get("mlp_ratio", 2.0)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            attn_dropout=float(cfg.model.get("attn_dropout", 0.0)),
            drop_path=float(cfg.model.get("drop_path", 0.1)),
            backbone_chunk_size=int(cfg.model.get("backbone_chunk_size", 28)),
        )
    if name == "videomae":
        from models.videomae import VideoMAEClassifier

        vm_frames = model_num_frames if model_num_frames > 0 else 16
        return VideoMAEClassifier(
            variant=str(cfg.model.get("variant", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            num_frames=vm_frames,
        )
    if name == "vjepa2":
        from models.vjepa2 import VJEPA2Classifier

        vj_frames = model_num_frames if model_num_frames > 0 else 16
        return VJEPA2Classifier(
            variant=str(
                cfg.model.get("variant", "facebook/vjepa2-vitl-fpc16-256-ssv2")
            ),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            ignore_mismatched_sizes=bool(
                cfg.model.get("ignore_mismatched_sizes", True)
            ),
            num_frames=vj_frames,
            input_size=int(cfg.model.get("input_size", 256)),
            dropout_p=float(cfg.model.get("dropout", 0.1)),
        )
    if name == "internvideo2":
        from models.internvideo import InternVideo2Classifier

        iv_frames = model_num_frames if model_num_frames > 0 else 8
        return InternVideo2Classifier(
            variant=str(
                cfg.model.get("variant", "OpenGVLab/InternVideo2-Stage2_1B-224p-f8")
            ),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            num_frames=iv_frames,
            input_size=int(cfg.model.get("input_size", 224)),
            dropout_p=float(cfg.model.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown model.name: {name}")
