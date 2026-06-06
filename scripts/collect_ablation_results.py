#!/usr/bin/env python3
"""Evaluate ablation checkpoints and write outputs/ablation/ablation_results.csv."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("track_a_tsm_resnet34", "track_a_cnn_lstm_improved")
AUGS = ("baseline", "rrc", "color_jitter", "erasing", "hflip", "mixup", "cutmix", "full")
OUT_CSV = ROOT / "outputs" / "ablation" / "ablation_results.csv"


def checkpoint_val_acc(ckpt: Path) -> float | None:
    import torch

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    val = raw.get("val_accuracy")
    return float(val) if val is not None else None


def run_evaluate(ckpt: Path) -> tuple[float | None, float | None]:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "evaluate.py"),
            f"training.checkpoint_path={ckpt}",
            "training.device=cuda",
            "training.num_workers=8",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(log, file=sys.stderr)
        return None, None

    top1_m = re.search(r"Top-1 accuracy:\s*([0-9.]+)", log)
    top5_m = re.search(r"Top-5 accuracy:\s*([0-9.]+)", log)
    top1 = float(top1_m.group(1)) if top1_m else None
    top5 = float(top5_m.group(1)) if top5_m else None
    return top1, top5


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | None]] = []

    for model in MODELS:
        for aug in AUGS:
            ckpt = ROOT / "outputs" / "ablation" / model / aug / "best_model.pt"
            if not ckpt.is_file():
                print(f"SKIP (missing): {ckpt}", file=sys.stderr)
                continue

            print(f"Evaluating {model} / {aug} ...", flush=True)
            val_ckpt = checkpoint_val_acc(ckpt)
            top1, top5 = run_evaluate(ckpt)
            rows.append(
                {
                    "model": model,
                    "aug": aug,
                    "checkpoint": str(ckpt),
                    "val_acc_checkpoint": val_ckpt,
                    "top1_eval": top1,
                    "top5_eval": top5,
                }
            )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "aug",
                "checkpoint",
                "val_acc_checkpoint",
                "top1_eval",
                "top5_eval",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    print(f"{'model':<30} {'aug':<14} {'val_ckpt':>10} {'top1':>8} {'top5':>8}")
    for row in rows:
        print(
            f"{row['model']:<30} {row['aug']:<14} "
            f"{row['val_acc_checkpoint']!s:>10} "
            f"{row['top1_eval']!s:>8} {row['top5_eval']!s:>8}"
        )


if __name__ == "__main__":
    main()
