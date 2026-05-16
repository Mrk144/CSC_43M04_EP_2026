"""
Mixture-of-experts sur **logits** : plusieurs checkpoints, même split val, top-1 / top-5.

Chaque expert est évalué comme dans ``evaluate.py`` (``num_frames`` et normalisation
tirés du checkpoint). Les logits sont ensuite combinés (moyenne, pondération, ou
poids appris sur le val).

Exemples (depuis ``src/``)::

    python moe_evaluate.py +moe=default
    python moe_evaluate.py +moe=default moe.combination=mean
    python moe_evaluate.py +moe=default moe.calibrate_temperature=false \\
        moe.experts='[{name:a,checkpoint:/path/a.pt},{name:b,checkpoint:/path/b.pt}]'

Combinations performantes en pratique : ``learned_ce`` + ``calibrate_temperature=true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from evaluate import load_model_from_checkpoint
from utils import build_transforms, set_seed


def _collect_logits(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (logits_all [N,C], labels_all [N]) on CPU float32."""
    logits_chunks: List[torch.Tensor] = []
    labels_chunks: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for video_batch, labels in data_loader:
            video_batch = video_batch.to(device)
            logits = model(video_batch)
            logits_chunks.append(logits.detach().float().cpu())
            labels_chunks.append(labels.long().cpu())
    if not logits_chunks:
        raise RuntimeError("Empty dataloader in _collect_logits")
    return torch.cat(logits_chunks, dim=0), torch.cat(labels_chunks, dim=0)


def _top1_top5(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, float]:
    logits = logits.to(labels.device)
    pred1 = logits.argmax(dim=1)
    top1 = (pred1 == labels).float().mean().item()
    _, top5 = logits.topk(5, dim=1, largest=True, sorted=True)
    top5_acc = top5.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
    return top1, top5_acc


def _find_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Scalar T>0 minimizing NLL of softmax(logits / T). Grid search (rapide, pas de dépendance scipy)."""
    best_t = 1.0
    best_ce = float("inf")
    for t in torch.linspace(0.5, 3.0, 26):
        ce = F.cross_entropy(logits / t, labels).item()
        if ce < best_ce:
            best_ce = ce
            best_t = float(t)
    return best_t


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-6)


def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.clamp(min=0)
    s = w.sum()
    if s <= 0:
        return torch.full_like(w, 1.0 / w.numel())
    return w / s


def _combine_mean(logits_stack: torch.Tensor) -> torch.Tensor:
    """logits_stack: (K, N, C)"""
    return logits_stack.mean(dim=0)


def _combine_val_acc_weights(
    logits_stack: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weight experts proportional to their individual top-1 accuracy (+ eps)."""
    k = logits_stack.size(0)
    scores = []
    for i in range(k):
        acc1, _ = _top1_top5(logits_stack[i], labels)
        scores.append(max(acc1, 1e-6))
    w = _normalize_weights(torch.tensor(scores, dtype=logits_stack.dtype))
    combined = (w.view(-1, 1, 1) * logits_stack).sum(dim=0)
    return combined, w


def _combine_weighted(
    logits_stack: torch.Tensor,
    weights: List[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = logits_stack.size(0)
    if len(weights) != k:
        raise ValueError(f"moe.weights length {len(weights)} != experts {k}")
    w = _normalize_weights(torch.tensor(weights, dtype=logits_stack.dtype))
    combined = (w.view(-1, 1, 1) * logits_stack).sum(dim=0)
    return combined, w


def _combine_learned_ce(
    logits_stack: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    lr: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax parametrization u -> w, minimize CE(combined logits, y). logits_stack: (K,N,C) on CPU."""
    k, n, c = logits_stack.shape
    l_dev = logits_stack.to(device)
    y = labels.to(device)

    u = torch.zeros(k, device=device, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(max(steps, 1)):
        opt.zero_grad()
        w = torch.softmax(u, dim=0)
        combined = (w.view(k, 1, 1) * l_dev).sum(dim=0)
        loss = F.cross_entropy(combined, y)
        loss.backward()
        opt.step()
    with torch.inference_mode():
        w = torch.softmax(u, dim=0)
        combined = (w.view(k, 1, 1) * l_dev).sum(dim=0)
    return combined.cpu(), w.detach().cpu()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    moe_cfg = cfg.get("moe")
    if moe_cfg is None:
        raise SystemExit("Add MoE config: python moe_evaluate.py +moe=default")
    print(OmegaConf.to_yaml(moe_cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    experts_cfg = list(moe_cfg.experts)
    if len(experts_cfg) < 2:
        raise SystemExit("moe.experts must list at least 2 checkpoints.")

    combination = str(moe_cfg.get("combination", "learned_ce"))
    calibrate = bool(moe_cfg.get("calibrate_temperature", True))

    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        val_samples = val_samples[: int(max_samples)]

    print(f"Validation videos: {len(val_samples)}")

    logits_list: List[torch.Tensor] = []
    expert_names: List[str] = []
    num_classes_ref: int | None = None

    for entry in experts_cfg:
        entry_d = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(entry_d, dict)
        ckpt_path = Path(str(entry_d["checkpoint"])).resolve()
        name = str(entry_d.get("name", ckpt_path.stem))
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Expert {name}: checkpoint not found: {ckpt_path}")

        raw: Dict[str, Any] = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
        saved_cfg = raw.get("config") or raw.get("cfg")
        if saved_cfg is None:
            raise ValueError(
                f"Expert {name}: checkpoint missing 'config' / 'cfg' (train with current train.py)."
            )
        nc = int(OmegaConf.create(saved_cfg).model.num_classes)
        if num_classes_ref is None:
            num_classes_ref = nc
        elif nc != num_classes_ref:
            raise ValueError(
                f"Expert {name}: num_classes={nc} != {num_classes_ref} (mixing head sizes is unsupported)."
            )

        model = load_model_from_checkpoint(raw, device)
        pretrained_used = bool(OmegaConf.create(saved_cfg).model.get("pretrained", False))
        eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained_used)
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

        logits, labels_ref = _collect_logits(model, val_loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if calibrate:
            t = _find_temperature(logits, labels_ref)
            print(f"  [{name}] temperature T={t:.3f}")
            logits = _apply_temperature(logits, t)

        acc1, acc5 = _top1_top5(logits, labels_ref)
        print(f"  [{name}] solo top-1={acc1:.4f} top-5={acc5:.4f}")

        logits_list.append(logits)
        expert_names.append(name)

    labels = labels_ref
    logits_stack = torch.stack(logits_list, dim=0)
    k = logits_stack.size(0)

    weights_used: torch.Tensor | None = None
    if combination == "mean":
        combined = _combine_mean(logits_stack)
        weights_used = torch.full((k,), 1.0 / k)
    elif combination == "val_acc":
        combined, weights_used = _combine_val_acc_weights(logits_stack, labels)
    elif combination == "weighted":
        wcfg = moe_cfg.get("weights")
        if wcfg is None:
            raise ValueError("moe.combination=weighted requires moe.weights as a list of length K.")
        combined, weights_used = _combine_weighted(logits_stack, [float(x) for x in list(wcfg)])
    elif combination == "learned_ce":
        print(
            "NOTE: learned_ce fits mixture weights on this validation set — "
            "the printed mixture accuracy is optimistically biased. "
            "For an unbiased score, hold out another split or use cross-val."
        )
        learned = moe_cfg.get("learned", {}) or {}
        steps = int(learned.get("steps", 400))
        lr = float(learned.get("lr", 0.07))
        comb_device = device if device.type == "cuda" else torch.device("cpu")
        combined, weights_used = _combine_learned_ce(
            logits_stack, labels, steps=steps, lr=lr, device=comb_device
        )
    else:
        raise ValueError(
            f"Unknown moe.combination={combination!r}; "
            f"expected mean | val_acc | weighted | learned_ce"
        )

    mix_top1, mix_top5 = _top1_top5(combined, labels)
    print("\n--- Mixture ---")
    if weights_used is not None:
        for i, name in enumerate(expert_names):
            print(f"  weight[{name}] = {weights_used[i].item():.4f}")
    print(f"Top-1 accuracy: {mix_top1:.4f}")
    print(f"Top-5 accuracy: {mix_top5:.4f}")

    solo_best_top1 = max(
        _top1_top5(logits_stack[i], labels)[0] for i in range(k)
    )
    print(f"\nBest expert top-1 (solo): {solo_best_top1:.4f}")
    if mix_top1 + 1e-4 < solo_best_top1:
        print(
            "Note: mixture top-1 is below the best solo model. "
            "Try dropping a weak expert, disable temperature, or use val_acc weights."
        )


if __name__ == "__main__":
    main()
