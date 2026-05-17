#!/usr/bin/env bash
# Generic two-phase trainer (linear probing -> full finetune).
#
# Usage (from repo root):
#   chmod +x scripts/train_two_phase.sh
#   ./scripts/train_two_phase.sh <model_tag>
#
# Where <model_tag> picks the pair of experiments
#   track_b_<model_tag>_head.yaml      (frozen backbone, head only)
#   track_b_<model_tag>_finetune.yaml  (full backbone, resumes phase 1)
#
# Supported tags out of the box:
#   videomae              VideoMAE-Base SSv2-finetuned (HuggingFace)
#   vjepa2                V-JEPA2 ViT-L (Meta)
#   internvideo2          InternVideo2 stage-2 1B (OpenGVLab, trust_remote_code)
#
# Optional overrides forwarded to BOTH phases:
#   TRAIN_EXTRA_ARGS="training.num_workers=8 dataset.max_samples=2000" \
#       ./scripts/train_two_phase.sh vjepa2

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_tag>" >&2
    echo "  e.g. $0 videomae | vjepa2 | internvideo2" >&2
    exit 1
fi

TAG="$1"
HEAD_EXP="track_b_${TAG}_head"
FT_EXP="track_b_${TAG}_finetune"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

if [[ ! -f "configs/experiment/${HEAD_EXP}.yaml" ]]; then
    echo "ERROR: missing config configs/experiment/${HEAD_EXP}.yaml" >&2
    exit 1
fi
if [[ ! -f "configs/experiment/${FT_EXP}.yaml" ]]; then
    echo "ERROR: missing config configs/experiment/${FT_EXP}.yaml" >&2
    exit 1
fi

OUTPUTS_DIR="$ROOT/outputs"
mkdir -p "$OUTPUTS_DIR"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
PHASE1_RUN_DIR="$OUTPUTS_DIR/${TAG}_phase1_${STAMP}"
PHASE2_RUN_DIR="$OUTPUTS_DIR/${TAG}_phase2_${STAMP}"

echo "=== PHASE 1 / 2: head only (linear probing) [$TAG] ==="
echo "Run dir: $PHASE1_RUN_DIR"
python train.py \
    experiment="$HEAD_EXP" \
    hydra.run.dir="$PHASE1_RUN_DIR" \
    ${TRAIN_EXTRA_ARGS:-}

PHASE1_CKPT="$PHASE1_RUN_DIR/best_model.pt"
if [[ ! -f "$PHASE1_CKPT" ]]; then
    echo "ERROR: phase-1 checkpoint not found at $PHASE1_CKPT" >&2
    exit 1
fi

echo "=== PHASE 2 / 2: full backbone finetune [$TAG] ==="
echo "Run dir: $PHASE2_RUN_DIR"
echo "Resuming model weights from: $PHASE1_CKPT"
python train.py \
    experiment="$FT_EXP" \
    hydra.run.dir="$PHASE2_RUN_DIR" \
    training.resume_from="$PHASE1_CKPT" \
    ${TRAIN_EXTRA_ARGS:-}

PHASE2_CKPT="$PHASE2_RUN_DIR/best_model.pt"
echo ""
echo "Both phases done for $TAG."
echo "  Phase 1 best:   $PHASE1_CKPT"
echo "  Phase 2 final:  $PHASE2_CKPT"
