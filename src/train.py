"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

from pathlib import Path
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
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_lstm_improved import CNNLSTMImproved
from models.cnn_transformer import CNNTransformer
from models.internvideo import InternVideo2Classifier
from models.pretrained_video import PretrainedVideoModel
from models.tsm_resnet import TSMResNet
from models.tsm_two_stream import TSMTwoStream
from models.videomae import VideoMAEClassifier
from models.vjepa2 import VJEPA2Classifier
from models.x3d import X3DClassifier
from utils import (
    build_transforms,
    class_weights_from_counts,
    count_class_frequencies,
    set_seed,
    split_train_val,
)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by ``cfg.model.name``.

    ``cfg.model.num_frames`` controls the *internal* temporal resolution the
    backbone operates on (Track A => 7, Track B => 16). The model itself
    interpolates the input from ``T_raw`` (whatever the dataset serves, 4 in
    our case) to that target inside its forward pass, so the saved ``.pt`` is
    self-contained.
    """
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained
    model_num_frames = int(cfg.model.get("num_frames", 0))

    if name == "cnn_baseline":
        return CNNBaseline(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=model_num_frames,
        )
    if name == "cnn_lstm":
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 256)),
            num_frames=model_num_frames,
        )
    if name == "cnn_lstm_improved":
        return CNNLSTMImproved(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 256)),
            dropout_p=float(cfg.model.get("dropout", 0.5)),
            num_frames=model_num_frames,
        )
    if name == "cnn_transformer":
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
        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        return TSMResNet(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
        )
    if name == "tsm_two_stream":
        tsm_frames = model_num_frames if model_num_frames > 0 else int(
            cfg.dataset.num_frames
        )
        return TSMTwoStream(
            num_classes=num_classes,
            num_frames=tsm_frames,
            pretrained=pretrained,
            dropout_p=float(cfg.model.get("dropout", 0.5)),
        )
    if name == "pretrained_video":
        pv_frames = model_num_frames if model_num_frames > 0 else 16
        return PretrainedVideoModel(
            backbone=str(cfg.model.backbone),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            num_frames=pv_frames,
        )
    if name == "videomae":
        vm_frames = model_num_frames if model_num_frames > 0 else 16
        return VideoMAEClassifier(
            variant=str(cfg.model.get("variant", "MCG-NJU/videomae-base-finetuned-ssv2")),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            num_frames=vm_frames,
        )
    if name == "vjepa2":
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
    if name == "x3d":
        x3d_frames = model_num_frames if model_num_frames > 0 else 0
        return X3DClassifier(
            variant=str(cfg.model.get("variant", "x3d_m")),
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
            num_frames=x3d_frames,
            input_size=int(cfg.model.get("input_size", 0)),
        )

    raise ValueError(f"Unknown model.name: {name}")


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

    loss_name = loss_cfg.get("name", "cross_entropy")
    if loss_name != "cross_entropy":
        print(
            "training.loss.class_weights is ignored when loss name is not "
            "cross_entropy (focal_loss has no class weighting yet)."
        )
        return None

    if spec == "auto":
        mode = str(loss_cfg.get("class_weights_mode", "inverse_freq"))
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
    """Pick the training loss from cfg.training.loss.

    Default: ``CrossEntropyLoss(label_smoothing=0.1)``. Setting
    ``training.loss.name = "focal_loss"`` switches to the focal variant.
    Optional ``class_weights`` are applied only for cross-entropy.
    """
    loss_cfg = cfg.training.get("loss", {}) or {}
    name = loss_cfg.get("name", "cross_entropy")
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))
    if name == "focal_loss":
        if class_weights is not None:
            print(
                "Ignoring class_weights: FocalLossWithSmoothing does not support "
                "per-class weights in this codebase."
            )
        gamma = float(loss_cfg.get("gamma", 2.0))
        return FocalLossWithSmoothing(smoothing=label_smoothing, gamma=gamma)
    ce_kwargs: Dict[str, Any] = {"label_smoothing": label_smoothing}
    if class_weights is not None:
        ce_kwargs["weight"] = class_weights
    return nn.CrossEntropyLoss(**ce_kwargs)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into two groups: with vs without weight decay.

    BatchNorm/LayerNorm parameters (rank-1 tensors) and biases get no weight
    decay; the rest gets full weight decay. Standard recipe to avoid hurting
    BN statistics and biases when training from scratch.
    """
    decay, no_decay = [], []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup over ``warmup_epochs`` then cosine annealing for the rest."""
    if warmup_epochs <= 0 or warmup_epochs >= total_epochs:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs, 1), eta_min=eta_min
        )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs * 5, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min
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
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
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

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(
        is_training=True, use_imagenet_norm=use_imagenet_norm
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
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

    opt_name = cfg.training.get("optimizer", {}).get("name", "adamw")
    lr = float(cfg.training.lr)
    wd = float(cfg.training.get("optimizer", {}).get("weight_decay", 0.0))
    param_groups = build_param_groups(model, weight_decay=wd)

    if opt_name == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr)
    elif opt_name == "sgd":
        momentum = float(cfg.training.get("optimizer", {}).get("momentum", 0.9))
        optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=True)
    else:
        optimizer = optim.Adam(param_groups, lr=lr)

    epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    scheduler = build_scheduler(
        optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs
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
            # For a full resume the architecture must match exactly: catch any
            # mismatch loudly instead of silently dropping params. For model-only
            # (two-phase) loads we keep ``strict=False`` because the new
            # classifier head intentionally differs from the pretrained shape.
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

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | lr {current_lr:.2e} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
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

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()