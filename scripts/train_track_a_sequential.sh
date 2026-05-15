#!/usr/bin/env bash
# Lance d'abord TSM single-stream, puis TSM two-stream (RGB + frame-diff).
# Le 2e run ne démarre que si le 1er se termine avec succès (exit 0).
#
# Usage (depuis la racine du dépôt) :
#   chmod +x scripts/train_track_a_sequential.sh
#   ./scripts/train_track_a_sequential.sh
#
# Overrides optionnels (exemple) :
#   TRAIN_EXTRA_ARGS="training.device=cpu dataset.max_samples=1000" ./scripts/train_track_a_sequential.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src"

echo "=== 1/2: experiment=track_a_best ==="
python train.py experiment=track_a_best ${TRAIN_EXTRA_ARGS:-}

echo "=== 2/2: experiment=track_a_two_stream ==="
python train.py experiment=track_a_two_stream ${TRAIN_EXTRA_ARGS:-}

echo "=== Terminé : les deux runs sont finis ==="
