from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
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
    torchvision v2 pipeline applied to the entire video tensor (T, C, H, W).

    - Train: random resized crop + mild photometric jitter + random grayscale +
      random erasing. NO horizontal flip / rotation because labels in this
      challenge are directional (e.g. left-to-right vs right-to-left).
    - Eval: deterministic shorter-side resize + center crop, matching the
      training crop size.
    """
    if use_imagenet_norm:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # Match the classic ImageNet recipe: resize shorter side to 256 then crop 224
    resize_size = int(round(image_size * 256 / 224))

    if is_training:
        return transforms.Compose([
            transforms.Lambda(lambda x: tv_tensors.Video(x)),
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
                antialias=True,
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])

    return transforms.Compose([
        transforms.Lambda(lambda x: tv_tensors.Video(x)),
        transforms.Resize(size=resize_size, antialias=True),
        transforms.CenterCrop(size=(image_size, image_size)),
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


def count_class_frequencies(
    samples: List[Tuple[Path, int]],
    num_classes: int,
) -> torch.Tensor:
    """Histogram of training labels (float64 tensor of shape [num_classes])."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _path, label in samples:
        if not (0 <= label < num_classes):
            raise ValueError(
                f"label {label} out of range [0, {num_classes}) for "
                f"{len(samples)} train samples"
            )
        counts[label] += 1.0
    return counts


def class_weights_from_counts(
    counts: torch.Tensor,
    mode: str = "inverse_freq",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Per-class weights for ``CrossEntropyLoss`` (mean-normalized float32 vector).

    - ``inverse_freq``: sklearn "balanced" style,
      w_c = N / (C * max(n_c, 1)) with N the sum of clamped counts.
    - ``effective_num``: Class-Balanced effective number (Cui et al.) on counts
      clamped to at least 1, then w_c proportional to 1 / E_c, mean-normalized.
    """
    if mode == "effective_num" and not (0.0 < beta < 1.0):
        raise ValueError(
            "class_weights beta must be in (0, 1) for effective_num mode"
        )

    counts = counts.to(dtype=torch.float64)
    safe = torch.clamp(counts, min=1.0)
    if mode == "inverse_freq":
        n = safe.sum()
        w = n / (float(safe.numel()) * safe)
    elif mode == "effective_num":
        b = torch.tensor(beta, dtype=torch.float64)
        eff_n = (1.0 - b.pow(safe)) / (1.0 - b)
        w = 1.0 / eff_n
    else:
        raise ValueError(
            f"Unknown class_weights_mode {mode!r}; "
            f"expected 'inverse_freq' or 'effective_num'"
        )
    w = w / w.mean()
    return w.float()
