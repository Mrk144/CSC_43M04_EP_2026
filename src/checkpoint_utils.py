"""Load a trained model from a checkpoint dict saved by train.py."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from train import build_model


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
