"""Load a trained model from a checkpoint dict saved by train.py."""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from models.vjepa2 import _is_ssv2_classification_variant
from train import build_model

# Track B experts: heavy VRAM, spatial resize inside the model, no TTA in MoE by default.
FOUNDATION_MODEL_NAMES = frozenset({"videomae", "vjepa2", "internvideo2"})


class ExpertEvalSettings(TypedDict):
    num_frames: int
    image_size: int
    batch_size: int
    model_name: str


def expert_eval_settings(
    raw: Dict[str, Any],
    cfg: DictConfig,
    expert_entry: Optional[Dict[str, Any]] = None,
) -> ExpertEvalSettings:
    """Infer dataloader / transform settings from a checkpoint (MoE-safe)."""
    saved = OmegaConf.create(raw.get("config") or raw.get("cfg") or {})
    model_cfg = saved.get("model") or {}
    model_name = str(raw.get("model_name") or model_cfg.get("name", ""))

    num_frames = int(
        raw.get(
            "num_frames",
            saved.get("dataset", {}).get("num_frames", cfg.dataset.num_frames),
        )
    )

    image_size = 224
    if model_name == "vjepa2":
        image_size = int(model_cfg.get("input_size", 256))
    elif model_name == "internvideo2":
        image_size = int(model_cfg.get("input_size", 224))

    batch_size = int(cfg.training.batch_size)
    if expert_entry is not None and expert_entry.get("batch_size") is not None:
        batch_size = int(expert_entry["batch_size"])
    elif model_name in FOUNDATION_MODEL_NAMES:
        saved_bs = saved.get("training", {}).get("batch_size")
        if saved_bs is not None:
            batch_size = int(saved_bs)
        else:
            batch_size = min(batch_size, 8)

    return {
        "num_frames": num_frames,
        "image_size": image_size,
        "batch_size": batch_size,
        "model_name": model_name,
    }


def aug_cfg_for_expert(cfg: DictConfig, model_name: str) -> DictConfig:
    """MoE: disable TTA for foundation models (2x forward, OOM risk)."""
    from utils import get_augmentation_cfg

    aug = get_augmentation_cfg(cfg)
    if model_name not in FOUNDATION_MODEL_NAMES:
        return aug
    tta = aug.get("tta")
    if tta is not None and bool(tta.get("enabled", False)):
        return OmegaConf.merge(aug, {"tta": {"enabled": False}})
    return aug


def _vjepa_checkpoint_is_legacy_encoder(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Old runs: ``backbone.*`` + ``classifier.*`` (encoder + linear head)."""
    keys = list(state_dict.keys())
    if any(k.startswith("backbone.") for k in keys):
        return True
    if any(k.startswith("model.") for k in keys):
        return False
    return False


def _cfg_for_checkpoint_load(
    cfg: DictConfig, state_dict: Dict[str, torch.Tensor]
) -> DictConfig:
    """Match architecture to checkpoint keys; skip HF download when weights are local."""
    cfg = OmegaConf.merge(cfg, {"model": {"pretrained": False}})
    if str(cfg.model.get("name", "")) != "vjepa2":
        return cfg

    if _vjepa_checkpoint_is_legacy_encoder(state_dict):
        variant = str(
            cfg.model.get("variant", "facebook/vjepa2-vitl-fpc16-256")
        )
        if _is_ssv2_classification_variant(variant):
            variant = "facebook/vjepa2-vitl-fpc16-256"
        cfg = OmegaConf.merge(
            cfg,
            {
                "model": {
                    "variant": variant,
                    "pretrained": False,
                }
            },
        )
        print(
            "V-JEPA2: loading legacy checkpoint (backbone + linear head).",
            flush=True,
        )
    else:
        print(
            "V-JEPA2: loading HuggingFace classification checkpoint (model.*).",
            flush=True,
        )
    return cfg


def _load_state_dict_compat(
    model: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    """Load weights; ignore extra keys (e.g. ``backbone.predictor``) when shapes differ."""
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != tensor.shape:
            skipped.append(key)
            continue
        filtered[key] = tensor

    missing = [k for k in model_state if k not in filtered]
    model.load_state_dict(filtered, strict=False)

    if skipped:
        preview = ", ".join(skipped[:6])
        extra = f" (+{len(skipped) - 6} more)" if len(skipped) > 6 else ""
        print(
            f"  Checkpoint keys not loaded ({len(skipped)}): {preview}{extra}",
            flush=True,
        )
    if missing:
        critical = [k for k in missing if "classifier" in k or "fc" in k]
        if critical:
            raise RuntimeError(
                "Checkpoint missing required head weights: "
                + ", ".join(critical[:8])
            )
        print(
            f"  Model submodules initialized fresh ({len(missing)} missing keys).",
            flush=True,
        )


def load_model_from_checkpoint(
    checkpoint: Dict[str, Any], device: torch.device
) -> torch.nn.Module:
    saved_cfg = checkpoint.get("config") or checkpoint.get("cfg")
    if saved_cfg is None:
        raise ValueError(
            "Checkpoint has no 'config' (or legacy 'cfg') entry. Train with the "
            "current train.py so the full Hydra config is saved with the weights."
        )
    state_dict = checkpoint["model_state_dict"]
    cfg = _cfg_for_checkpoint_load(OmegaConf.create(saved_cfg), state_dict)
    model = build_model(cfg)
    _load_state_dict_compat(model, state_dict)
    model.to(device)
    model.eval()
    return model
