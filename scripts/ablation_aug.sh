#!/usr/bin/env bash
# Ablation: TSM-ResNet34 & CNN-LSTM, 5 epochs, one augmentation at a time.
#
# Usage (from repo root):
#   chmod +x scripts/ablation_aug.sh
#   ./scripts/ablation_aug.sh
#
# Optional:
#   TRAIN_EXTRA_ARGS="training.device=cuda training.num_workers=8" ./scripts/ablation_aug.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS=(track_a_tsm_resnet34 track_a_cnn_lstm_improved)
AUGS=(baseline rrc color_jitter erasing hflip mixup cutmix full)

aug_overrides() {
    local aug="$1"
    case "$aug" in
        baseline)
            echo "augmentation=ablation_base"
            ;;
        rrc)
            echo "augmentation=ablation_base augmentation.random_resized_crop.enabled=true augmentation.random_resized_crop.scale=[0.7,1.0]"
            ;;
        color_jitter)
            echo "augmentation=ablation_base augmentation.color_jitter.enabled=true"
            ;;
        erasing)
            echo "augmentation=ablation_base augmentation.random_erasing.enabled=true augmentation.random_erasing.prob=0.25"
            ;;
        hflip)
            echo "augmentation=ablation_base augmentation.hflip.enabled=true augmentation.hflip.prob=0.5"
            ;;
        mixup)
            echo "augmentation=ablation_base augmentation.mixup.enabled=true augmentation.mixup.alpha=0.2 augmentation.mixup.prob=0.15"
            ;;
        cutmix)
            echo "augmentation=ablation_base augmentation.cutmix.enabled=true augmentation.cutmix.alpha=1.0 augmentation.cutmix.prob=0.15"
            ;;
        full)
            echo "augmentation=default"
            ;;
        *)
            echo "Unknown aug: $aug" >&2
            exit 1
            ;;
    esac
}

for MODEL in "${MODELS[@]}"; do
    for AUG in "${AUGS[@]}"; do
        RUN_DIR="$ROOT/outputs/ablation/${MODEL}/${AUG}"
        echo ""
        echo "=== ${MODEL} / ${AUG} ==="
        echo "Run dir: $RUN_DIR"

        # shellcheck disable=SC2046
        python src/train.py \
            experiment="$MODEL" \
            training.epochs=5 \
            hydra.run.dir="$RUN_DIR" \
            $(aug_overrides "$AUG") \
            ${TRAIN_EXTRA_ARGS:-}
    done
done

echo ""
echo "Ablation training done. Checkpoints under outputs/ablation/<model>/<aug>/best_model.pt"
