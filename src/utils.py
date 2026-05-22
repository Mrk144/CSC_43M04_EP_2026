from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

# Pulling left-to-right <-> pulling right-to-left (dataset class indices).
HFLIP_LABEL_SWAP: Dict[int, int] = {18: 19, 19: 18}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    aug_cfg: object | None = None,
) -> transforms.Compose:
    """
    torchvision v2 pipeline applied to the entire video tensor (T, C, H, W).

    Pixels are scaled to [0, 1] then normalized with ImageNet mean/std.

    - Train: random resized crop, color jitter, random erasing (configurable).
      Horizontal flip is **not** here: it runs in ``VideoFrameDataset`` with
      label swap for classes 18 <-> 19 (left/right pull).
    - Eval: deterministic shorter-side resize + center crop.
    """
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    # Match the classic ImageNet recipe: resize shorter side to 256 then crop 224
    resize_size = int(round(image_size * 256 / 224))

    use_rrc = True
    use_jitter = True
    use_erase = True
    jitter_brightness = 0.2
    jitter_contrast = 0.2
    jitter_saturation = 0.2
    jitter_hue = 0.05
    erase_p = 0.25
    rrc_scale = (0.7, 1.0)

    if aug_cfg is not None and is_training:
        def _on(key: str, default: bool = True) -> bool:
            block = aug_cfg.get(key) if hasattr(aug_cfg, "get") else None
            if block is None:
                return default
            if isinstance(block, bool):
                return bool(block)
            return bool(block.get("enabled", default))

        use_rrc = _on("random_resized_crop")
        use_jitter = _on("color_jitter")
        use_erase = _on("random_erasing")
        if use_jitter:
            jitter_brightness = float(
                aug_cfg.get("color_jitter", {}).get("brightness", jitter_brightness)
            )
            jitter_contrast = float(
                aug_cfg.get("color_jitter", {}).get("contrast", jitter_contrast)
            )
            jitter_saturation = float(
                aug_cfg.get("color_jitter", {}).get("saturation", jitter_saturation)
            )
            jitter_hue = float(aug_cfg.get("color_jitter", {}).get("hue", jitter_hue))
        if use_erase:
            erase_p = float(aug_cfg.get("random_erasing", {}).get("prob", erase_p))
        if use_rrc:
            rrc_scale = tuple(aug_cfg.get("random_resized_crop", {}).get(
                "scale", list(rrc_scale)
            ))

    if is_training:
        steps = [transforms.Lambda(lambda x: tv_tensors.Video(x))]
        if use_rrc:
            steps.append(
                transforms.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=rrc_scale,
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                    antialias=True,
                )
            )
        else:
            steps.extend([
                transforms.Resize(size=resize_size, antialias=True),
                transforms.CenterCrop(size=(image_size, image_size)),
            ])
        if use_jitter:
            steps.append(
                transforms.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                )
            )
        steps.append(transforms.RandomGrayscale(p=0.1))
        steps.append(transforms.ToDtype(torch.float32, scale=True))
        steps.append(transforms.Normalize(mean=mean, std=std))
        if use_erase:
            steps.append(transforms.RandomErasing(p=erase_p, scale=(0.02, 0.2)))
        return transforms.Compose(steps)

    return transforms.Compose([
        transforms.Lambda(lambda x: tv_tensors.Video(x)),
        transforms.Resize(size=resize_size, antialias=True),
        transforms.CenterCrop(size=(image_size, image_size)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ])


def swap_directional_label(label: int) -> int:
    return HFLIP_LABEL_SWAP.get(int(label), int(label))


def swap_directional_logits(logits: torch.Tensor) -> torch.Tensor:
    """Swap class dimensions 18 and 19 (for TTA after a horizontal flip)."""
    if logits.size(-1) <= max(HFLIP_LABEL_SWAP):
        return logits
    out = logits.clone()
    out[..., 18], out[..., 19] = logits[..., 19].clone(), logits[..., 18].clone()
    return out


def get_augmentation_cfg(cfg: DictConfig) -> DictConfig:
    if cfg.get("augmentation") is not None:
        return cfg.augmentation
    return OmegaConf.create({})


def _cfg_bool(aug: DictConfig, key: str, default: bool = True) -> bool:
    block = aug.get(key)
    if block is None:
        return default
    if isinstance(block, bool):
        return bool(block)
    return bool(block.get("enabled", default))


def _cfg_float(aug: DictConfig, *keys: str, default: float) -> float:
    node: object = aug
    for k in keys:
        if node is None or not hasattr(node, "get"):
            return default
        node = node.get(k)
    return float(node) if node is not None else default


def apply_horizontal_flip_video(
    video: torch.Tensor,
    label: int,
    *,
    prob: float,
) -> Tuple[torch.Tensor, int]:
    """Flip width dimension; swap labels 18 <-> 19 when mirrored."""
    if prob <= 0.0 or random.random() >= prob:
        return video, label
    flipped = torch.flip(video, dims=[-1])
    return flipped, swap_directional_label(label)


def _sample_mix_lambda(alpha: float) -> float:
    alpha = max(alpha, 1e-6)
    return float(torch.distributions.Beta(alpha, alpha).sample().item())


def mixup_batch(
    video: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns mixed video and soft targets (B, C)."""
    if video.size(0) < 2:
        return video, _labels_to_soft(labels, num_classes, label_smoothing)

    lam = _sample_mix_lambda(alpha)
    index = torch.randperm(video.size(0), device=video.device)
    mixed = lam * video + (1.0 - lam) * video[index]
    soft_a = _labels_to_soft(labels, num_classes, label_smoothing)
    soft_b = _labels_to_soft(labels[index], num_classes, label_smoothing)
    return mixed, lam * soft_a + (1.0 - lam) * soft_b


def cutmix_batch(
    video: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix on spatial dims; same box for all frames in each clip."""
    b, _t, _c, h, w = video.shape
    if b < 2:
        return video, _labels_to_soft(labels, num_classes, label_smoothing)

    lam = _sample_mix_lambda(alpha)
    index = torch.randperm(b, device=video.device)

    cut_rat = torch.sqrt(torch.tensor(1.0 - lam, device=video.device))
    cut_w = int((w * cut_rat).item())
    cut_h = int((h * cut_rat).item())

    cx = int(torch.randint(0, w, (1,), device=video.device).item())
    cy = int(torch.randint(0, h, (1,), device=video.device).item())
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, x1 + cut_w)
    y2 = min(h, y1 + cut_h)

    mixed = video.clone()
    mixed[:, :, :, y1:y2, x1:x2] = video[index, :, :, y1:y2, x1:x2]

    box_area = float((x2 - x1) * (y2 - y1))
    lam_adj = 1.0 - box_area / float(w * h)
    lam_adj = max(0.0, min(1.0, lam_adj))

    soft_a = _labels_to_soft(labels, num_classes, label_smoothing)
    soft_b = _labels_to_soft(labels[index], num_classes, label_smoothing)
    return mixed, lam_adj * soft_a + (1.0 - lam_adj) * soft_b


def _labels_to_soft(
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
) -> torch.Tensor:
    soft = F.one_hot(labels.long(), num_classes=num_classes).float()
    if label_smoothing > 0.0:
        soft = soft * (1.0 - label_smoothing) + label_smoothing / num_classes
    return soft


def maybe_batch_augment(
    video: torch.Tensor,
    labels: torch.Tensor,
    aug_cfg: DictConfig,
    *,
    num_classes: int,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply cutmix or mixup. Returns (video, hard_labels, soft_targets|None)."""
    if not _cfg_bool(aug_cfg, "enabled", default=True):
        return video, labels, None

    cutmix_on = _cfg_bool(aug_cfg, "cutmix", default=False)
    mixup_on = _cfg_bool(aug_cfg, "mixup", default=False)
    cut_prob = _cfg_float(aug_cfg, "cutmix", "prob", default=0.0)
    mix_prob = _cfg_float(aug_cfg, "mixup", "prob", default=0.0)

    r = random.random()
    if cutmix_on and r < cut_prob:
        mixed, soft = cutmix_batch(
            video,
            labels,
            alpha=_cfg_float(aug_cfg, "cutmix", "alpha", default=1.0),
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )
        return mixed, labels, soft

    if mixup_on and r < mix_prob:
        mixed, soft = mixup_batch(
            video,
            labels,
            alpha=_cfg_float(aug_cfg, "mixup", "alpha", default=0.2),
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )
        return mixed, labels, soft

    return video, labels, None


def soft_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    if class_weights is not None:
        hard = soft_targets.argmax(dim=-1)
        loss = loss * class_weights[hard]
    return loss.mean()


def compute_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    soft_targets: Optional[torch.Tensor],
) -> torch.Tensor:
    if soft_targets is None:
        return loss_fn(logits, labels)
    weight = getattr(loss_fn, "weight", None)
    # Mixup/cutmix: soft CE (focal modulation is not applied on soft targets).
    return soft_cross_entropy(logits, soft_targets, class_weights=weight)


@torch.no_grad()
def forward_logits_with_tta(
    model: nn.Module,
    video: torch.Tensor,
    aug_cfg: DictConfig,
    *,
    use_autocast: bool = False,
) -> torch.Tensor:
    """Optional TTA: average logits with hflip branch (swap cls 18/19 on flip)."""
    tta = aug_cfg.get("tta") or {}
    if not bool(tta.get("enabled", False)):
        if use_autocast and video.is_cuda:
            with torch.amp.autocast("cuda"):
                return model(video)
        return model(video)

    views = [video]
    if bool(tta.get("hflip", True)):
        views.append(torch.flip(video, dims=[-1]))

    logits_sum = None
    for i, view in enumerate(views):
        if use_autocast and view.is_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(view)
        else:
            logits = model(view)
        if i == 1 and bool(tta.get("swap_directional_logits", True)):
            logits = swap_directional_logits(logits)
        logits_sum = logits if logits_sum is None else logits_sum + logits

    assert logits_sum is not None
    return logits_sum / len(views)


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
    max_weight_ratio: float = 50.0,
) -> torch.Tensor:
    """Per-class weights for ``CrossEntropyLoss`` (mean-normalized float32 vector).

    Only classes with ``counts > 0`` are reweighted. Absent classes get weight **0**
    because with ``label_smoothing > 0`` PyTorch scales *each* smoothed class term
    by ``weight[c]``; a large weight on a never-seen class (e.g. 27) forces the model
    to predict that class only.

    Weights are clipped to ``[mean/max_weight_ratio, mean*max_weight_ratio]`` on
    present classes only.

    - ``inverse_freq``: sklearn "balanced" style on present classes only,
      w_c = N / (C_present * n_c).
    - ``effective_num``: Class-Balanced effective number (Cui et al.).
    """
    if mode == "effective_num" and not (0.0 < beta < 1.0):
        raise ValueError(
            "class_weights beta must be in (0, 1) for effective_num mode"
        )

    counts = counts.to(dtype=torch.float64)
    present = counts > 0
    if not bool(present.any()):
        return torch.ones_like(counts, dtype=torch.float32)

    n_present = int(present.sum().item())
    c = counts[present]
    if mode == "inverse_freq":
        n = c.sum()
        w_pos = n / (float(n_present) * c)
    elif mode == "effective_num":
        b = torch.tensor(beta, dtype=torch.float64)
        eff_n = (1.0 - b.pow(c)) / (1.0 - b)
        w_pos = 1.0 / eff_n
    else:
        raise ValueError(
            f"Unknown class_weights_mode {mode!r}; "
            f"expected 'inverse_freq' or 'effective_num'"
        )

    w = torch.zeros(counts.numel(), dtype=torch.float64)
    w[present] = w_pos
    w[present] = w[present] / w[present].mean()
    if max_weight_ratio > 1.0:
        mean_p = w[present].mean()
        lo = mean_p / max_weight_ratio
        hi = mean_p * max_weight_ratio
        w[present] = w[present].clamp(min=lo, max=hi)
        w[present] = w[present] / w[present].mean()
    return w.float()
