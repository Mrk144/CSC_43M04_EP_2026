#!/usr/bin/env bash
# Enchaîne 4 modèles Track A, 30 epochs chacun (même recettes que les YAML d’exp, sauf epochs).
#
# Depuis la racine du dépôt :
#   chmod +x scripts/train_track_a_four_30ep.sh
#   ./scripts/train_track_a_four_30ep.sh
#
# Variables optionnelles :
#   EPOCHS=30              (défaut 30)
#   TRAIN_EXTRA_ARGS="training.batch_size=32 dataset.max_samples=500"
#
# Sorties Hydra : src/outputs/<date>_* (un dossier par lancement).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/src"

EPOCHS="${EPOCHS:-30}"

echo "=== CNN-LSTM improved ($EPOCHS ep) ==="
python train.py experiment=track_a_cnn_lstm_improved training.epochs="$EPOCHS" ${TRAIN_EXTRA_ARGS:-}

echo "=== CNN-Transformer ($EPOCHS ep) ==="
python train.py experiment=track_a_cnn_transformer training.epochs="$EPOCHS" ${TRAIN_EXTRA_ARGS:-}

echo "=== TSM ResNet (track_a_best) ($EPOCHS ep) ==="
python train.py experiment=track_a_best training.epochs="$EPOCHS" ${TRAIN_EXTRA_ARGS:-}

echo "=== TSM two-stream ($EPOCHS ep) ==="
python train.py experiment=track_a_two_stream training.epochs="$EPOCHS" ${TRAIN_EXTRA_ARGS:-}

echo "Done. Check src/outputs/ for run directories and best_model.pt in each."
