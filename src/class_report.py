"""
Per-class precision / recall, global accuracy, and row-normalized confusion.

``P(predicted=j | true=i) = C[i,j] / sum_j C[i,j]`` (percent split of errors when
true class is i). Same validation setup as ``evaluate.py``.

Run from ``src/``::

    python class_report.py +report=default
    python class_report.py +report=default report.output_dir=./report_out \\
        'report.models=[{name:m1,checkpoint:/abs/c1.pt}]'

**Recall** for class i = fraction of true-i samples predicted as i (class-wise
correct rate). **Precision** for class i = fraction of predictions i that are
truly i.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from evaluate import load_model_from_checkpoint
from utils import build_transforms, set_seed


def _build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """C[i,j] = count with true label i and predicted j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if y_true.size > 0:
        np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def _per_class_precision_recall(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    return precision, recall, support


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm.astype(np.float64),
        support,
        out=np.zeros_like(cm, dtype=np.float64),
        where=support > 0,
    )


def _collect_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    y_true_list: List[torch.Tensor] = []
    y_pred_list: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for video_batch, labels in data_loader:
            video_batch = video_batch.to(device)
            logits = model(video_batch)
            pred = logits.argmax(dim=1)
            y_true_list.append(labels.detach().cpu())
            y_pred_list.append(pred.detach().cpu())
    if not y_true_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return (
        torch.cat(y_true_list).numpy().astype(np.int64),
        torch.cat(y_pred_list).numpy().astype(np.int64),
    )


def _print_metrics_table(
    precision: np.ndarray,
    recall: np.ndarray,
    support: np.ndarray,
    cm: np.ndarray,
) -> None:
    total = float(cm.sum())
    acc = float(np.trace(cm)) / total if total > 0 else 0.0
    print(f"\nGlobal top-1 accuracy: {acc:.4f}  (N={int(total)})")

    mask = support > 0
    macro_p = float(precision[mask].mean()) if mask.any() else 0.0
    macro_r = float(recall[mask].mean()) if mask.any() else 0.0
    w_support = support / max(total, 1.0)
    weighted_p = float((precision * w_support).sum()) if total > 0 else 0.0
    weighted_r = float((recall * w_support).sum()) if total > 0 else 0.0

    print(
        f"Macro avg precision/recall (classes with support): {macro_p:.4f} / {macro_r:.4f}"
    )
    print(
        f"Weighted avg precision/recall (by true-class support): {weighted_p:.4f} / {weighted_r:.4f}"
    )
    print("")
    print(f"{'class':>5}  {'support':>8}  {'precision':>9}  {'recall':>7}")
    print("-" * 40)
    for c in range(len(support)):
        print(
            f"{c:5d}  {int(support[c]):8d}  {precision[c]:9.4f}  {recall[c]:7.4f}"
        )


def _print_top_confusions(row_norm: np.ndarray, top_k: int) -> None:
    n = row_norm.shape[0]
    print("\nTop confusions P(pred=j | true=i), excluding j=i:")
    for i in range(n):
        row = row_norm[i]
        order = np.argsort(-row)
        shown = 0
        for j in order:
            if j == i:
                continue
            if row[j] <= 0:
                break
            pct = 100.0 * row[j]
            print(f"  true={i} -> pred={j}: {pct:.2f}%")
            shown += 1
            if shown >= top_k:
                break


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]


def _export_csvs(
    out_dir: Path,
    basename: str,
    precision: np.ndarray,
    recall: np.ndarray,
    support: np.ndarray,
    cm: np.ndarray,
    row_norm: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_path = out_dir / f"precision_recall_{basename}.csv"
    with pr_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "support", "precision", "recall"])
        for c in range(len(support)):
            w.writerow([c, int(support[c]), precision[c], recall[c]])

    raw_path = out_dir / f"confusion_counts_{basename}.csv"
    num_classes = cm.shape[0]
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_pred"] + [str(j) for j in range(num_classes)])
        for i in range(num_classes):
            w.writerow([str(i)] + [str(int(cm[i, j])) for j in range(num_classes)])

    norm_path = out_dir / f"confusion_row_norm_{basename}.csv"
    with norm_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_pred"] + [str(j) for j in range(num_classes)])
        for i in range(num_classes):
            w.writerow([str(i)] + [f"{row_norm[i, j]:.8f}" for j in range(num_classes)])
    print(f"Wrote CSV under {out_dir}")


def _run_one_model(
    cfg: DictConfig,
    report_cfg: DictConfig,
    entry: Dict[str, Any],
    device: torch.device,
) -> None:
    ckpt_path = Path(str(entry["checkpoint"])).resolve()
    name = str(entry.get("name", ckpt_path.stem))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for {name}: {ckpt_path}")

    raw: Dict[str, Any] = torch.load(
        ckpt_path, map_location=device, weights_only=False
    )
    saved_cfg = OmegaConf.create(raw.get("config") or raw.get("cfg"))
    num_classes = int(saved_cfg.model.num_classes)

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

    y_true, y_pred = _collect_predictions(model, val_loader, device)
    cm = _build_confusion_matrix(y_true, y_pred, num_classes)
    precision, recall, support = _per_class_precision_recall(cm)
    row_norm = _row_normalize(cm)

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"Checkpoint: {ckpt_path}")
    print("Definitions: recall = P(pred=i | true=i); precision = P(true=i | pred=i)")
    _print_metrics_table(precision, recall, support, cm)
    top_k = int(report_cfg.get("top_confusions", 5))
    _print_top_confusions(row_norm, top_k)

    out = report_cfg.get("output_dir")
    if out is not None and str(out).strip():
        out_dir = Path(str(out)).resolve()
        _export_csvs(
            out_dir,
            _safe_filename(name),
            precision,
            recall,
            support,
            cm,
            row_norm,
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    report_cfg = cfg.get("report")
    if report_cfg is None:
        raise SystemExit(
            "Missing report config. Example: python class_report.py +report=default"
        )
    print(OmegaConf.to_yaml(report_cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    models_list = list(report_cfg.get("models", []))
    if len(models_list) == 0:
        raise SystemExit("report.models is empty; add at least one checkpoint.")

    for entry in models_list:
        entry_d = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(entry_d, dict)
        _run_one_model(cfg, report_cfg, entry_d, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
