"""
Mixture-of-experts sur **logits** : plusieurs checkpoints, même split val, top-1 / top-5.

Combinaisons supportées : ``mean`` | ``val_acc``.

Depuis la racine du dépôt ::

    python src/moe_evaluate.py +moe=default
    python src/moe_evaluate.py +moe=default moe.combination=mean
    python src/evaluate.py +moe=default
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from moe_core import run_moe_evaluation
from utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("moe") is None:
        raise SystemExit("Add MoE config: python moe_evaluate.py +moe=default")
    set_seed(int(cfg.dataset.seed))
    run_moe_evaluation(cfg)


if __name__ == "__main__":
    main()
