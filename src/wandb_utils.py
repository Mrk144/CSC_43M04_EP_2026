"""Optional Weights & Biases logging for train.py and evaluate.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def _wandb_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.get("wandb") or OmegaConf.create({"enabled": False})


def setup_wandb(
    cfg: DictConfig,
    run_dir: Path,
    model: Optional[Any] = None,
    *,
    job_type: str | None = None,
    name_prefix: str = "",
) -> Any:
    """Start a W&B run if ``wandb.enabled`` is true. Returns the run or None."""
    wb = _wandb_cfg(cfg)
    if not bool(wb.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError:
        print(
            "wandb.enabled=true but package 'wandb' is not installed. "
            "Run: uv add wandb"
        )
        return None

    name = wb.get("name")
    if not name:
        prefix = str(wb.get("name_prefix", name_prefix) or "")
        name = f"{prefix}{cfg.model.name}_{run_dir.name}"

    tags = list(wb.get("tags") or [])
    if cfg.get("experiment"):
        tags.append(str(cfg.experiment))

    init_kw: Dict[str, Any] = {
        "project": str(wb.get("project", "csc-what-happens-next")),
        "name": str(name),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dir": str(run_dir),
        "reinit": True,
    }
    entity = wb.get("entity")
    if entity:
        init_kw["entity"] = str(entity)
    if tags:
        init_kw["tags"] = tags
    notes = wb.get("notes")
    if notes:
        init_kw["notes"] = str(notes)
    mode = wb.get("mode")
    if mode:
        init_kw["mode"] = str(mode)
    resolved_job_type = job_type or wb.get("job_type")
    if resolved_job_type:
        init_kw["job_type"] = str(resolved_job_type)

    run = wandb.init(**init_kw)

    if model is not None and bool(wb.get("watch", False)):
        log_freq = int(wb.get("watch_log_freq", 100))
        wandb.watch(model, log=str(wb.get("watch_log", "gradients")), log_freq=log_freq)

    print(f"W&B run: {run.url}")
    return run


def log_epoch(
    cfg: DictConfig,
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    best_val_accuracy: float,
    is_best: bool,
) -> None:
    """Log scalar metrics at the end of an epoch."""
    if not bool(_wandb_cfg(cfg).get("enabled", False)):
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    step = epoch + 1
    wandb.log(
        {
            "epoch": step,
            "lr": lr,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/best_accuracy": best_val_accuracy,
            "val/is_best": int(is_best),
        },
        step=step,
    )
    wandb.run.summary["best_val_accuracy"] = best_val_accuracy


def log_eval_results(cfg: DictConfig, metrics: Dict[str, Any]) -> None:
    """Log evaluation metrics (single model or MoE mixture)."""
    if not bool(_wandb_cfg(cfg).get("enabled", False)):
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    log_payload: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            log_payload[key] = value
            wandb.run.summary[key] = value
        elif isinstance(value, str):
            wandb.run.summary[key] = value

    if log_payload:
        wandb.log(log_payload)


def finish_wandb(cfg: DictConfig) -> None:
    if not bool(_wandb_cfg(cfg).get("enabled", False)):
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
