"""
Evaluate on the **full** validation split (``dataset.val_dir``).

**Un modèle** (checkpoint unique) ::

    python src/evaluate.py training.checkpoint_path=outputs/.../best_model.pt

**MoE** (plusieurs checkpoints, combinaison mean ou val_acc) ::

    python src/evaluate.py +moe=default
    python src/evaluate.py +moe=default moe.combination=mean
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from checkpoint_utils import load_model_from_checkpoint
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from moe_core import run_moe_evaluation
from utils import build_transforms, set_seed


def _evaluate_single_model(cfg: DictConfig, device: torch.device) -> None:
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    raw: Dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = load_model_from_checkpoint(raw, device)

    pretrained_used = bool(raw.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    num_frames = int(raw.get("num_frames", cfg.dataset.num_frames))

    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for video_batch, labels in val_loader:
            video_batch = video_batch.to(device)
            labels = labels.to(device)
            logits = model(video_batch)

            predictions_top1 = logits.argmax(dim=1)
            correct_top1 += int((predictions_top1 == labels).sum().item())

            _, predictions_top5 = logits.topk(5, dim=1, largest=True, sorted=True)
            matches_top5 = predictions_top5.eq(labels.view(-1, 1)).any(dim=1)
            correct_top5 += int(matches_top5.sum().item())

            total += labels.size(0)

    top1_accuracy = correct_top1 / max(total, 1)
    top5_accuracy = correct_top5 / max(total, 1)

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 accuracy: {top5_accuracy:.4f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    moe_cfg = cfg.get("moe")
    if moe_cfg is not None and moe_cfg.get("experts"):
        run_moe_evaluation(cfg)
        return

    _evaluate_single_model(cfg, device)


if __name__ == "__main__":
    main()
