#!/usr/bin/env bash
# Two-phase VideoMAE training: head-only linear probing, then full finetune.
#
# Usage (from repo root):
#   chmod +x scripts/train_videomae_two_phase.sh
#   ./scripts/train_videomae_two_phase.sh
#
# Optional overrides forwarded to BOTH phases:
#   TRAIN_EXTRA_ARGS="training.num_workers=8 dataset.max_samples=2000" \
#       ./scripts/train_videomae_two_phase.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

OUTPUTS_DIR="$ROOT/outputs"
mkdir -p "$OUTPUTS_DIR"

PHASE1_RUN_DIR="$OUTPUTS_DIR/videomae_phase1_$(date +%Y-%m-%d_%H-%M-%S)"
PHASE2_RUN_DIR="$OUTPUTS_DIR/videomae_phase2_$(date +%Y-%m-%d_%H-%M-%S)"

echo "=== PHASE 1 / 2: head only (linear probing) ==="
echo "Run dir: $PHASE1_RUN_DIR"
python train.py \
    experiment=track_b_videomae_head \
    hydra.run.dir="$PHASE1_RUN_DIR" \
    ${TRAIN_EXTRA_ARGS:-}

PHASE1_CKPT="$PHASE1_RUN_DIR/best_model.pt"
if [[ ! -f "$PHASE1_CKPT" ]]; then
    echo "ERROR: phase-1 checkpoint not found at $PHASE1_CKPT" >&2
    exit 1
fi

echo "=== PHASE 2 / 2: full backbone finetune ==="
echo "Run dir: $PHASE2_RUN_DIR"
echo "Resuming model weights from: $PHASE1_CKPT"
python train.py \
    experiment=track_b_videomae_finetune \
    hydra.run.dir="$PHASE2_RUN_DIR" \
    training.resume_from="$PHASE1_CKPT" \
    ${TRAIN_EXTRA_ARGS:-}

PHASE2_CKPT="$PHASE2_RUN_DIR/best_model.pt"
echo ""
echo "Both phases done."
echo "  Phase 1 best:   $PHASE1_CKPT"
echo "  Phase 2 final:  $PHASE2_CKPT"
