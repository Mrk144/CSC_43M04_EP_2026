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

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from utils import build_transforms, set_seed, split_train_val

# ==============================================================================
# --- NOUVEAUX IMPORTS (TRACK A) ---
# ==============================================================================
import torch.optim as optim
from models.cnn_lstm_improved import CNNLSTMImproved
from losses import FocalLossWithSmoothing
from models.tsm_resnet import TSMResNet
from hydra.core.hydra_config import HydraConfig
# ==============================================================================


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    elif name == "cnn_lstm_improved":
        return CNNLSTMImproved(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(cfg.model.get("lstm_hidden_size", 512)),
            dropout_p=float(cfg.model.get("dropout", 0.5))
        )
    elif name == "tsm_resnet":
        return TSMResNet(
            num_classes=num_classes,
            num_frames=int(cfg.dataset.num_frames),
            pretrained=pretrained
        )
    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    # ==========================================================================
    # === NOUVEAU PARAMÈTRE (AMP) : Le Scaler ===
    # ==========================================================================
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ======================================================================
        # === NOUVEAU CODE (AMP) : Forward Pass avec Autocast ===
        # Convertit automatiquement les opérations compatibles en 16-bit
        # ======================================================================
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(video_batch)  # (B, num_classes)
            loss = loss_fn(logits, labels)

        # ======================================================================
        # === NOUVEAU CODE (AMP) : Backward Pass avec Scaler ===
        # Protège les gradients contre l'underflow (devenir trop petits)
        # ======================================================================
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # ======================================================================

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

        # ======================================================================
        # === NOUVEAU CODE (AMP) : Autocast pour l'Évaluation ===
        # Accélère l'inférence en mode évaluation
        # ======================================================================
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)
        # ======================================================================

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

    model = build_model(cfg).to(device)

    # --- Initialisation de la Loss ---
    loss_name = cfg.training.get("loss", {}).get("name", "cross_entropy")
    if loss_name == "focal_loss":
        smoothing = float(cfg.training.get("loss", {}).get("label_smoothing", 0.1))
        gamma = float(cfg.training.get("loss", {}).get("gamma", 2.0))
        loss_fn = FocalLossWithSmoothing(smoothing=smoothing, gamma=gamma)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # --- Initialisation de l'Optimiseur ---
    opt_name = cfg.training.get("optimizer", {}).get("name", "adam")
    lr = float(cfg.training.lr)
    wd = float(cfg.training.get("optimizer", {}).get("weight_decay", 0.0))
    
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = int(cfg.training.epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ==========================================================================
    # === NOUVEAU CODE (AMP) : Initialisation du GradScaler ===
    # Ne s'active que si un GPU (cuda) est utilisé
    # ==========================================================================
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    # ==========================================================================

    best_val_accuracy = 0.0
    # Récupère le chemin du dossier créé par Hydra (ex: outputs/2026-05-14_16-30-00)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    
    # Prépare le nom du fichier (best_model.pt)
    checkpoint_path = Path(cfg.training.checkpoint_path)
    
    # Fusionne les deux : si le chemin n'est pas absolu, on le met dans le dossier de run
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path

    for epoch in range(int(cfg.training.epochs)):
        # ======================================================================
        # === NOUVEAU CODE (AMP) : On passe le scaler à train_one_epoch ===
        # ======================================================================
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler
        )
        # ======================================================================
        
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if "lstm" in cfg.model.name:
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()