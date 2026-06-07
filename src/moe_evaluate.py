from __future__ import annotations
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from moe_core import run_moe_evaluation
from utils import set_seed
from wandb_utils import finish_wandb, setup_wandb

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.get('moe') is None:
        raise SystemExit('Add MoE config: python moe_evaluate.py +moe=default')
    set_seed(int(cfg.dataset.seed))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    setup_wandb(cfg, run_dir, job_type='eval', name_prefix='eval_')
    try:
        run_moe_evaluation(cfg)
    finally:
        finish_wandb(cfg)
if __name__ == '__main__':
    main()
