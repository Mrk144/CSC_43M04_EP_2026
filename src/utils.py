from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
# Utilisation de la v2 pour la gestion des tenseurs vidéo (T, C, H, W)
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
) -> transforms.Compose:
    """
    Pipeline torchvision v2 pour garantir la cohérence temporelle sur 4 frames.
    """
    # --- Gestion de la normalisation ---
    if use_imagenet_norm:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    if is_training:
        return transforms.Compose([
            # ==================================================================
            # === PRÉPARATION VIDÉO ===
            # Reçoit le tenseur (T, C, H, W) venant de video_dataset.py
            # ==================================================================
            transforms.Lambda(lambda x: tv_tensors.Video(x)),
            
            # ==================================================================
            # === AUGMENTATIONS GÉOMÉTRIQUES SÉCURISÉES ===
            # S'appliquent IDENTIQUEMENT aux 4 frames
            # ==================================================================
            transforms.RandomResizedCrop(size=(image_size, image_size), scale=(0.95, 1.0), antialias=True),
            transforms.RandomRotation(degrees=5),

            # ==================================================================
            # === AUGMENTATIONS VISUELLES ===
            # Gardent la même intensité sur toute la séquence
            # ==================================================================
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

            # ==================================================================
            # === CONVERSION ET NORMALISATION ===
            # Obligatoire : ToDtype remplace ToTensor pour les tenseurs
            # ==================================================================
            transforms.ToDtype(torch.float32, scale=True), 
            transforms.Normalize(mean=mean, std=std),

        ])
    else:
        # --- Validation ---
        return transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
        ])

@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    max_k = max(topk)
    batch_size = targets.size(0)

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)

def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples