"""Dump softmax predictions of a checkpoint on the val and test splits.

Used to build ensembles: instead of re-running model inference every time we
tweak ensembling weights, we cache each model's softmax probabilities in npz
files. ``src/ensemble.py`` then combines them in numpy with zero GPU cost.

Outputs:
    <run_dir>/preds/val.npz   { video_names, softmax (N, C), labels }
    <run_dir>/preds/test.npz  { video_names, softmax (N, C) }

Example (from src/):
    python predict_softmax.py training.checkpoint_path=/abs/best_model.pt \
        predict.output_dir=/abs/preds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from create_submission import (
    _index_video_folders,
    build_model_from_checkpoint,
    discover_all_test_videos,
)
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed


def _resolve_output_dir(cfg: DictConfig, checkpoint_path: Path) -> Path:
    """Where to write val.npz/test.npz. Defaults next to the checkpoint."""
    override = cfg.get("predict", {}).get("output_dir") if "predict" in cfg else None
    if override:
        return Path(str(override)).resolve()
    return (checkpoint_path.parent / "preds").resolve()


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (softmax (N, C), labels (N,))."""
    model.eval()
    softmaxes: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for video_batch, lbl in loader:
        video_batch = video_batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(video_batch)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        softmaxes.append(probs)
        labels.append(lbl.cpu().numpy())
    softmax = np.concatenate(softmaxes, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    if softmax.shape[1] != n_classes:
        print(
            f"Warning: model output dim {softmax.shape[1]} != expected {n_classes}"
        )
    return softmax, labels_arr


def _video_names_from_samples(samples: List[Tuple[Path, int]]) -> List[str]:
    return [path.name for path, _ in samples]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.pretrained))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained)
    num_classes = int(ckpt.get("num_classes", cfg.model.num_classes))

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)

    output_dir = _resolve_output_dir(cfg, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Predictions will be written to: {output_dir}", flush=True)

    # --- Validation split ----------------------------------------------------
    val_dir = Path(cfg.dataset.val_dir).resolve()
    print(f"Indexing validation videos under: {val_dir}", flush=True)
    val_samples = collect_video_samples(val_dir)
    val_names = _video_names_from_samples(val_samples)
    val_dataset = VideoFrameDataset(
        root_dir=val_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Running inference on {len(val_dataset)} validation clips", flush=True)
    val_softmax, val_labels = _run_inference(model, val_loader, device, num_classes)
    val_pred = val_softmax.argmax(axis=1)
    val_acc = float((val_pred == val_labels).mean())
    print(f"Validation top-1 accuracy: {val_acc:.4f}")

    val_out = output_dir / "val.npz"
    np.savez(
        val_out,
        video_names=np.array(val_names),
        softmax=val_softmax.astype(np.float32),
        labels=val_labels.astype(np.int64),
        val_accuracy=np.float32(val_acc),
        model_name=str(ckpt.get("model_name", "unknown")),
    )
    print(f"Wrote {val_out}")

    # --- Test split ----------------------------------------------------------
    test_dir = Path(cfg.dataset.test_dir).resolve()
    print(f"Indexing test videos under: {test_dir}", flush=True)
    manifest_cfg = cfg.dataset.get("test_manifest")
    if manifest_cfg:
        from create_submission import load_manifest_video_names, resolve_video_dirs

        manifest_path = Path(str(manifest_cfg)).resolve()
        test_names = load_manifest_video_names(manifest_path)
        test_dirs = resolve_video_dirs(test_dir, test_names)
    else:
        test_names, test_dirs = discover_all_test_videos(test_dir)
    test_samples: List[Tuple[Path, int]] = [(p, 0) for p in test_dirs]

    test_dataset = VideoFrameDataset(
        root_dir=test_dir,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=test_samples,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Running inference on {len(test_dataset)} test clips", flush=True)
    test_softmax, _ = _run_inference(model, test_loader, device, num_classes)

    test_out = output_dir / "test.npz"
    np.savez(
        test_out,
        video_names=np.array(test_names),
        softmax=test_softmax.astype(np.float32),
        model_name=str(ckpt.get("model_name", "unknown")),
    )
    print(f"Wrote {test_out}")

    # We also leave a small companion file with summary metadata for humans.
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(
        f"model_name={ckpt.get('model_name', 'unknown')}\n"
        f"checkpoint={checkpoint_path}\n"
        f"val_clips={len(val_dataset)}  val_accuracy={val_acc:.4f}\n"
        f"test_clips={len(test_dataset)}\n"
        f"num_frames={num_frames}\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
