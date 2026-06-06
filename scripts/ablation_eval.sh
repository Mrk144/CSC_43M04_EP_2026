#!/usr/bin/env bash
# Evaluate all ablation checkpoints and write a comparison CSV.
#
# Usage (from repo root):
#   chmod +x scripts/ablation_eval.sh
#   ./scripts/ablation_eval.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
python scripts/collect_ablation_results.py
