"""Load a trained model from a checkpoint dict saved by train.py."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import build_model

if TYPE_CHECKING:
    from omegaconf import DictConfig


def infer_use_imagenet_norm(checkpoint: Dict[str, Any], cfg: "DictConfig") -> bool:
    """Match train.py eval transforms (saved flag or Hydra default)."""
    if "use_imagenet_norm" in checkpoint:
        return bool(checkpoint["use_imagenet_norm"])
    saved = checkpoint.get("config") or {}
    if isinstance(saved, dict):
        model_cfg = saved.get("model")
        if isinstance(model_cfg, dict) and model_cfg.get("use_imagenet_norm") is not None:
            return bool(model_cfg["use_imagenet_norm"])
    return bool(cfg.model.get("use_imagenet_norm", True))


def load_model_from_checkpoint(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    saved_cfg = checkpoint.get("config") or checkpoint.get("cfg")
    if saved_cfg is None:
        raise ValueError(
            "Checkpoint has no 'config' (or legacy 'cfg') entry. Train with the "
            "current train.py so the full Hydra config is saved with the weights."
        )
    cfg = OmegaConf.create(saved_cfg)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
