"""
Train a video classifier on folders of frames.

From the repository root::

    python src/train.py
    python src/train.py experiment=track_a_best

Or the same via the launcher (no ``.py``)::

    python src/train experiment=track_a_best

From ``src/`` (legacy)::

    python train.py experiment=track_a_best
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_CONFIG_DIR = str(_SRC_DIR / "configs")

from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import torch.optim as optim
from hydra.core.hydra_config import HydraConfig

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from losses import FocalLossWithSmoothing
from model_factory import build_model
from utils import (
    build_transforms,
    class_weights_from_counts,
    compute_loss,
    count_class_frequencies,
    get_augmentation_cfg,
    maybe_batch_augment,
    set_seed,
    split_train_val,
)
from wandb_utils import finish_wandb, log_epoch, setup_wandb


def _resolve_class_weights_for_training(
    cfg: DictConfig,
    train_samples: list[tuple[Path, int]],
    num_classes: int,
) -> torch.Tensor | None:
    """Build optional per-class weights from ``cfg.training.loss.class_weights``."""
    loss_cfg = cfg.training.get("loss", {}) or {}
    spec = loss_cfg.get("class_weights", None)
    if spec is None:
        return None

    if spec == "auto":
        mode = str(loss_cfg.get("class_weights_mode", "effective_num"))
        beta = float(loss_cfg.get("class_weights_beta", 0.9999))
        counts = count_class_frequencies(train_samples, num_classes)
        weights = class_weights_from_counts(counts, mode=mode, beta=beta)
        zeros = int((counts == 0).sum().item())
        print(
            f"Class weights (auto, mode={mode}): "
            f"min={weights.min().item():.4f} max={weights.max().item():.4f}; "
            f"classes with 0 train samples: {zeros}"
        )
        print(f"Per-class train counts: {counts.long().tolist()}")
        return weights

    if OmegaConf.is_list(spec) or isinstance(spec, (list, tuple)):
        w = torch.tensor(list(spec), dtype=torch.float32)
    else:
        raise ValueError(
            f"training.loss.class_weights must be null, 'auto', or a list; "
            f"got {spec!r}"
        )
    if w.numel() != num_classes:
        raise ValueError(
            f"class_weights length {w.numel()} != num_classes {num_classes}"
        )
    return w


def build_loss(
    cfg: DictConfig,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Pick the training loss from cfg.training.loss."""
    loss_cfg = cfg.training.get("loss", {}) or {}
    name = loss_cfg.get("name", "focal_loss")
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))
    if name == "focal_loss":
        gamma = float(loss_cfg.get("gamma", 2.0))
        return FocalLossWithSmoothing(
            smoothing=label_smoothing,
            gamma=gamma,
            class_weights=class_weights,
        )
    ce_kwargs: Dict[str, Any] = {"label_smoothing": label_smoothing}
    if class_weights is not None:
        ce_kwargs["weight"] = class_weights
    return nn.CrossEntropyLoss(**ce_kwargs)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay / no-decay groups for AdamW."""
    skip_names: set[str] = set()
    no_weight_decay_fn = getattr(model, "no_weight_decay", None)
    if callable(no_weight_decay_fn):
        skip_names = set(no_weight_decay_fn())

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if param.ndim <= 1 or leaf in skip_names:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _resolve_scheduler_eta_min(
    scheduler_cfg: Dict[str, Any],
    lr: float,
    *,
    default: float = 1e-6,
) -> float:
    ratio = scheduler_cfg.get("eta_min_ratio")
    if ratio is not None:
        return float(lr) * float(ratio)
    return float(scheduler_cfg.get("eta_min", default))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    *,
    lr: float,
    scheduler_cfg: Dict[str, Any] | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup over ``warmup_epochs`` then cosine annealing for the rest."""
    scheduler_cfg = scheduler_cfg or {}
    eta_min = _resolve_scheduler_eta_min(scheduler_cfg, lr)
    t_max_ratio = float(scheduler_cfg.get("t_max_ratio", 1.0))

    if warmup_epochs <= 0 or warmup_epochs >= total_epochs:
        cosine_t_max = max(1, int(max(total_epochs, 1) * t_max_ratio))
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=eta_min
        )

    post_warmup = max(total_epochs - warmup_epochs, 1)
    cosine_t_max = max(1, int(post_warmup * t_max_ratio))
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs * 5, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=eta_min
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    *,
    aug_cfg: DictConfig | None = None,
    num_classes: int = 33,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    aug_cfg = aug_cfg or OmegaConf.create({})

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        video_batch, labels, soft_targets = maybe_batch_augment(
            video_batch,
            labels,
            aug_cfg,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(video_batch)
            loss = compute_loss(loss_fn, logits, labels, soft_targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def log_prediction_snapshot(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    tag: str = "val",
) -> None:
    """Print argmax class histogram on one batch (debug collapse / label issues)."""
    model.eval()
    video_batch, labels = next(iter(data_loader))
    video_batch = video_batch.to(device)
    labels = labels.to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(video_batch)
    preds = logits.argmax(dim=1)
    counts = torch.bincount(preds, minlength=logits.size(-1))
    top = counts.topk(min(5, counts.numel()))
    top_str = ", ".join(
        f"cls {i.item()}:{c.item()}" for i, c in zip(top.indices, top.values) if c > 0
    )
    match = float((preds == labels).float().mean().item())
    print(
        f"[{tag} batch] top preds: {top_str or 'none'} | "
        f"batch acc {match:.3f} | labels [{int(labels.min())}, {int(labels.max())}]"
    )
    model.train()


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    aug_cfg = get_augmentation_cfg(cfg)
    train_transform = build_transforms(is_training=True, aug_cfg=aug_cfg)
    eval_transform = build_transforms(is_training=False, aug_cfg=aug_cfg)

    hflip_prob = 0.0
    hflip_block = aug_cfg.get("hflip")
    if hflip_block is not None and bool(hflip_block.get("enabled", False)):
        hflip_prob = float(hflip_block.get("prob", 0.5))

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
        hflip_prob=hflip_prob,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(cfg.model.num_classes)
    class_weights_tensor = _resolve_class_weights_for_training(
        cfg, train_samples, num_classes
    )

    model = build_model(cfg).to(device)

    loss_fn = build_loss(cfg, class_weights_tensor)
    loss_fn = loss_fn.to(device)
    loss_cfg = cfg.training.get("loss", {}) or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))

    opt_name = cfg.training.get("optimizer", {}).get("name", "adamw")
    lr = float(cfg.training.lr)
    wd = float(cfg.training.get("optimizer", {}).get("weight_decay", 0.05))
    param_groups = build_param_groups(model, weight_decay=wd)

    if opt_name == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr)
    elif opt_name == "sgd":
        momentum = float(cfg.training.get("optimizer", {}).get("momentum", 0.9))
        optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        optimizer = optim.Adam(param_groups, lr=lr)

    epochs = int(cfg.training.get("epochs", 40))
    warmup_epochs = int(cfg.training.get("warmup_epochs", 3))
    scheduler_cfg = cfg.training.get("scheduler", {}) or {}
    scheduler = build_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        lr=lr,
        scheduler_cfg=OmegaConf.to_container(scheduler_cfg, resolve=True)
        if scheduler_cfg
        else {},
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_accuracy = 0.0
    resume_path = cfg.training.get("resume_from")
    resume_model_only = bool(cfg.training.get("resume_model_only", False))

    if resume_path:
        resume_path = Path(resume_path).resolve()
        if resume_path.exists():
            mode = "model only" if resume_model_only else "full state"
            print(f"Resuming training from ({mode}): {resume_path}")
            checkpoint = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            model.load_state_dict(
                checkpoint["model_state_dict"], strict=not resume_model_only
            )
            if not resume_model_only:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_accuracy = checkpoint.get("val_accuracy", 0.0)
                print(
                    f"Resumed at epoch {start_epoch} "
                    f"(best val acc so far: {best_val_accuracy:.4f})"
                )
            else:
                print(
                    "Loaded model weights only. Optimizer/scheduler/epoch "
                    "reset for a fresh phase."
                )
        else:
            print(f"Resume checkpoint not found: {resume_path}")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_path = Path(cfg.training.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be written to: {checkpoint_path}")

    setup_wandb(cfg, run_dir, model=model)

    if start_epoch == 0:
        log_prediction_snapshot(model, val_loader, device, tag="val init")

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler,
            aug_cfg=aug_cfg,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        if epoch == start_epoch:
            log_prediction_snapshot(model, val_loader, device, tag="val")

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "model_num_frames": int(cfg.model.get("num_frames", 0)),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if "lstm" in cfg.model.name:
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 256)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

        log_epoch(
            cfg,
            epoch=epoch,
            total_epochs=int(cfg.training.epochs),
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
            best_val_accuracy=best_val_accuracy,
            is_best=is_best,
        )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    finish_wandb(cfg)


if __name__ == "__main__":
    main()
