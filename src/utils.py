"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms


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
    Standard torchvision pipeline for single RGB frames.
    """
    # ==================================================================
    # --- CODE ORIGINAL : Gestion de la normalisation ---
    # ==================================================================
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        return transforms.Compose([
            # ==================================================================
            # === AUGMENTATIONS GÉOMÉTRIQUES SÉCURISÉES ===
            # Protègent les classes 018, 019, 008, 009 (mouvements directionnels)
            # ==================================================================
            
            # Zoom et recadrage (original)
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            
            # Micro-rotation (original)
            transforms.RandomRotation(degrees=5),

            # --- NOUVEAUTÉ 1 : Translation (Random Affine) ---
            # Déplace l'image sans la retourner. Aide à la robustesse si la main 
            # n'est pas parfaitement centrée.
            #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

            # --- NOUVEAUTÉ 2 : Perspective légère ---
            # Simule un changement d'angle de la caméra par rapport à la table.
            #transforms.RandomPerspective(distortion_scale=0.2, p=0.4),

            # ==================================================================
            # === AUGMENTATIONS VISUELLES ET DE QUALITÉ ===
            # Simulent des variations d'éclairage et de capteur
            # ==================================================================
            
            # Modification des couleurs (original)
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

            # --- NOUVEAUTÉ 3 : Netteté (Sharpness) ---
            # Rend l'image plus ou moins "piquée" pour simuler différentes caméras.
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            
            # Flou gaussien (original)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.2),

            # --- NOUVEAUTÉ 4 : Posterisation ---
            # Réduit la palette de couleurs. Force le modèle à voir les formes 
            # globales plutôt que les textures fines.
            #transforms.RandomApply([
            #    transforms.RandomPosterize(bits=4)], p=0.2),

            # ==================================================================
            # --- CONVERSION ET RÉGULARISATION FINALE ---
            # ==================================================================
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # ==================================================================
        # --- CODE ORIGINAL : Validation ---
        # ==================================================================
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k correctness for each k in topk.
    """
    # --- CODE ORIGINAL : Calcul des métriques ---
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
    """
    Shuffle then split a list into train and validation portions.
    """
    # --- CODE ORIGINAL : Séparation Train / Val ---
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