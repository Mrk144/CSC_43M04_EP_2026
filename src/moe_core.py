from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from checkpoint_utils import aug_cfg_for_expert, expert_eval_settings, load_model_from_checkpoint
from utils import build_transforms, forward_logits_with_tta
from wandb_utils import log_eval_results

def collect_logits(model: nn.Module, data_loader: DataLoader, device: torch.device, aug_cfg: DictConfig | None=None) -> Tuple[torch.Tensor, torch.Tensor]:
    aug_cfg = aug_cfg or OmegaConf.create({})
    logits_chunks: List[torch.Tensor] = []
    labels_chunks: List[torch.Tensor] = []
    model.eval()
    use_autocast = device.type == 'cuda'
    with torch.inference_mode():
        for (video_batch, labels) in data_loader:
            video_batch = video_batch.to(device)
            logits = forward_logits_with_tta(model, video_batch, aug_cfg, use_autocast=use_autocast)
            logits_chunks.append(logits.detach().float().cpu())
            labels_chunks.append(labels.long().cpu())
    if not logits_chunks:
        raise RuntimeError('Empty dataloader in collect_logits')
    return (torch.cat(logits_chunks, dim=0), torch.cat(labels_chunks, dim=0))

def top1_top5(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    logits = logits.to(labels.device)
    pred1 = logits.argmax(dim=1)
    top1 = (pred1 == labels).float().mean().item()
    (_, top5) = logits.topk(5, dim=1, largest=True, sorted=True)
    top5_acc = top5.eq(labels.view(-1, 1)).any(dim=1).float().mean().item()
    return (top1, top5_acc)

def find_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_t = 1.0
    best_ce = float('inf')
    for t in torch.linspace(0.5, 3.0, 26):
        ce = F.cross_entropy(logits / t, labels).item()
        if ce < best_ce:
            best_ce = ce
            best_t = float(t)
    return best_t

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / max(temperature, 1e-06)

def normalize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.clamp(min=0)
    s = w.sum()
    if s <= 0:
        return torch.full_like(w, 1.0 / w.numel())
    return w / s

def combine_mean(logits_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    k = logits_stack.size(0)
    return (logits_stack.mean(dim=0), torch.full((k,), 1.0 / k))

def combine_val_acc_weights(logits_stack: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    k = logits_stack.size(0)
    scores = []
    for i in range(k):
        (acc1, _) = top1_top5(logits_stack[i], labels)
        scores.append(max(acc1, 1e-06))
    w = normalize_weights(torch.tensor(scores, dtype=logits_stack.dtype))
    combined = (w.view(-1, 1, 1) * logits_stack).sum(dim=0)
    return (combined, w)

def combine_logits(logits_stack: torch.Tensor, labels: torch.Tensor, combination: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if combination == 'mean':
        return combine_mean(logits_stack)
    if combination == 'val_acc':
        return combine_val_acc_weights(logits_stack, labels)
    raise ValueError(f'Unknown moe.combination={combination!r}; use mean | val_acc')

def run_moe_evaluation(cfg: DictConfig) -> None:
    moe_cfg = cfg.moe
    print(OmegaConf.to_yaml(moe_cfg))
    device_str = cfg.training.device
    if device_str == 'cuda' and (not torch.cuda.is_available()):
        print('CUDA not available; using CPU.')
        device_str = 'cpu'
    device = torch.device(device_str)
    experts_cfg = list(moe_cfg.experts)
    if len(experts_cfg) < 2:
        raise SystemExit('moe.experts must list at least 2 checkpoints.')
    combination = str(moe_cfg.get('combination', 'val_acc'))
    calibrate = bool(moe_cfg.get('calibrate_temperature', True))
    val_dir = Path(cfg.dataset.val_dir).resolve()
    val_samples = collect_video_samples(val_dir)
    max_samples = cfg.dataset.get('max_samples')
    if max_samples is not None:
        val_samples = val_samples[:int(max_samples)]
    print(f'Validation videos: {len(val_samples)}')
    print(f'Combination: {combination}')
    logits_list: List[torch.Tensor] = []
    expert_names: List[str] = []
    num_classes_ref: int | None = None
    labels_ref: torch.Tensor | None = None
    for entry in experts_cfg:
        entry_d = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(entry_d, dict)
        ckpt_path = Path(str(entry_d['checkpoint'])).resolve()
        name = str(entry_d.get('name', ckpt_path.stem))
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'Expert {name}: checkpoint not found: {ckpt_path}')
        raw: Dict[str, Any] = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        saved_cfg = raw.get('config') or raw.get('cfg')
        if saved_cfg is None:
            raise ValueError(f"Expert {name}: checkpoint missing 'config' / 'cfg' (train with current train.py).")
        nc = int(OmegaConf.create(saved_cfg).model.num_classes)
        if num_classes_ref is None:
            num_classes_ref = nc
        elif nc != num_classes_ref:
            raise ValueError(f'Expert {name}: num_classes={nc} != {num_classes_ref} (mixing head sizes is unsupported).')
        settings = expert_eval_settings(raw, cfg, entry_d)
        print(f"  [{name}] {settings['model_name']} | T={settings['num_frames']} | {settings['image_size']}px | bs={settings['batch_size']}", flush=True)
        model = load_model_from_checkpoint(raw, device)
        eval_transform = build_transforms(is_training=False, image_size=settings['image_size'])
        aug_cfg = aug_cfg_for_expert(cfg, settings['model_name'])
        val_dataset = VideoFrameDataset(root_dir=val_dir, num_frames=settings['num_frames'], transform=eval_transform, sample_list=val_samples)
        val_loader = DataLoader(val_dataset, batch_size=settings['batch_size'], shuffle=False, num_workers=int(cfg.training.num_workers), pin_memory=device.type == 'cuda')
        (logits, labels_ref) = collect_logits(model, val_loader, device, aug_cfg)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if calibrate:
            t = find_temperature(logits, labels_ref)
            print(f'  [{name}] temperature T={t:.3f}')
            logits = apply_temperature(logits, t)
        (acc1, acc5) = top1_top5(logits, labels_ref)
        print(f'  [{name}] solo top-1={acc1:.4f} top-5={acc5:.4f}')
        logits_list.append(logits)
        expert_names.append(name)
    assert labels_ref is not None
    logits_stack = torch.stack(logits_list, dim=0)
    k = logits_stack.size(0)
    (combined, weights_used) = combine_logits(logits_stack, labels_ref, combination)
    (mix_top1, mix_top5) = top1_top5(combined, labels_ref)
    print('\n--- Mixture ---')
    for (i, name) in enumerate(expert_names):
        print(f'  weight[{name}] = {weights_used[i].item():.4f}')
    print(f'Top-1 accuracy: {mix_top1:.4f}')
    print(f'Top-5 accuracy: {mix_top5:.4f}')
    solo_best_top1 = max((top1_top5(logits_stack[i], labels_ref)[0] for i in range(k)))
    print(f'\nBest expert top-1 (solo): {solo_best_top1:.4f}')
    metrics: Dict[str, Any] = {'eval/top1_accuracy': mix_top1, 'eval/top5_accuracy': mix_top5, 'eval/num_samples': len(val_samples), 'eval/solo_best_top1': solo_best_top1, 'eval/moe_combination': combination}
    for (i, name) in enumerate(expert_names):
        metrics[f'eval/moe_weight/{name}'] = float(weights_used[i].item())
        (acc1, acc5) = top1_top5(logits_stack[i], labels_ref)
        metrics[f'eval/expert_top1/{name}'] = acc1
        metrics[f'eval/expert_top5/{name}'] = acc5
    log_eval_results(cfg, metrics)

def _expert_logits_on_samples(cfg: DictConfig, ckpt_path: Path, name: str, samples: List[Tuple[Path, int]], data_root: Path, device: torch.device, *, expert_entry: Dict[str, Any] | None=None, calibrate: bool=False, calib_samples: List[Tuple[Path, int]] | None=None, fixed_temperature: float | None=None) -> torch.Tensor:
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Expert {name}: checkpoint not found: {ckpt_path}')
    raw: Dict[str, Any] = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_cfg = raw.get('config') or raw.get('cfg')
    if saved_cfg is None:
        raise ValueError(f"Expert {name}: checkpoint missing 'config' / 'cfg' (train with current train.py).")
    settings = expert_eval_settings(raw, cfg, expert_entry)
    print(f"  [{name}] {settings['model_name']} | T={settings['num_frames']} | {settings['image_size']}px | bs={settings['batch_size']}", flush=True)
    model = load_model_from_checkpoint(raw, device)
    eval_transform = build_transforms(is_training=False, image_size=settings['image_size'])
    aug_cfg = aug_cfg_for_expert(cfg, settings['model_name'])

    def _loader(sample_list: List[Tuple[Path, int]]) -> DataLoader:
        dataset = VideoFrameDataset(root_dir=data_root, num_frames=settings['num_frames'], transform=eval_transform, sample_list=sample_list)
        return DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False, num_workers=int(cfg.training.num_workers), pin_memory=device.type == 'cuda')
    temperature = fixed_temperature
    if calibrate and calib_samples and (temperature is None):
        calib_loader = _loader(calib_samples)
        (calib_logits, calib_labels) = collect_logits(model, calib_loader, device, aug_cfg)
        temperature = find_temperature(calib_logits, calib_labels)
        print(f'  [{name}] temperature T={temperature:.3f} (from val)', flush=True)
    infer_loader = _loader(samples)
    (logits, _) = collect_logits(model, infer_loader, device, aug_cfg)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if temperature is not None:
        logits = apply_temperature(logits, temperature)
    return logits

def run_moe_submission(cfg: DictConfig, video_dirs: List[Path], data_root: Path, device: torch.device) -> List[int]:
    moe_cfg = cfg.moe
    experts_cfg = list(moe_cfg.experts)
    if len(experts_cfg) < 2:
        raise SystemExit('moe.experts must list at least 2 checkpoints.')
    combination = str(moe_cfg.get('combination', 'mean'))
    if combination not in ('mean', 'val_acc'):
        raise ValueError(f'Unknown moe.combination={combination!r}; use mean | val_acc')
    calibrate = bool(moe_cfg.get('calibrate_temperature', True))
    need_val = calibrate or combination == 'val_acc'
    val_samples: List[Tuple[Path, int]] | None = None
    if need_val:
        val_dir = Path(cfg.dataset.val_dir).resolve()
        val_samples = collect_video_samples(val_dir)
        max_samples = cfg.dataset.get('max_samples')
        if max_samples is not None:
            val_samples = val_samples[:int(max_samples)]
        parts = []
        if calibrate:
            parts.append('temperature calibration')
        if combination == 'val_acc':
            parts.append('val_acc weights')
        print(f"Val pass ({', '.join(parts)}): {len(val_samples)} clips", flush=True)
    test_samples: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    print(f'MoE submission: {len(test_samples)} test clips, combination={combination}', flush=True)
    test_logits_list: List[torch.Tensor] = []
    val_logits_list: List[torch.Tensor] = []
    expert_names: List[str] = []
    val_labels_ref: torch.Tensor | None = None
    num_classes_ref: int | None = None
    for entry in experts_cfg:
        entry_d = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(entry_d, dict)
        ckpt_path = Path(str(entry_d['checkpoint'])).resolve()
        name = str(entry_d.get('name', ckpt_path.stem))
        raw: Dict[str, Any] = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        saved_cfg = raw.get('config') or raw.get('cfg')
        if saved_cfg is None:
            raise ValueError(f'Expert {name}: checkpoint missing config.')
        nc = int(OmegaConf.create(saved_cfg).model.num_classes)
        if num_classes_ref is None:
            num_classes_ref = nc
        elif nc != num_classes_ref:
            raise ValueError(f'Expert {name}: num_classes={nc} != {num_classes_ref}')
        val_dir = Path(cfg.dataset.val_dir).resolve()
        temperature: float | None = None
        if need_val and val_samples:
            print(f'  [{name}] val pass...', flush=True)
            val_logits = _expert_logits_on_samples(cfg, ckpt_path, name, val_samples, val_dir, device, expert_entry=entry_d, calibrate=False)
            if val_labels_ref is None:
                val_labels_ref = torch.tensor([lab for (_, lab) in val_samples], dtype=torch.long)
            if calibrate:
                temperature = find_temperature(val_logits, val_labels_ref)
                print(f'  [{name}] temperature T={temperature:.3f} (from val)', flush=True)
                val_logits = apply_temperature(val_logits, temperature)
            if combination == 'val_acc':
                val_logits_list.append(val_logits)
        print(f'  [{name}] test pass...', flush=True)
        if calibrate and need_val and val_samples:
            test_logits = _expert_logits_on_samples(cfg, ckpt_path, name, test_samples, data_root, device, expert_entry=entry_d, fixed_temperature=temperature)
        else:
            test_logits = _expert_logits_on_samples(cfg, ckpt_path, name, test_samples, data_root, device, expert_entry=entry_d, calibrate=calibrate, calib_samples=val_samples if calibrate else None)
        test_logits_list.append(test_logits)
        expert_names.append(name)
    test_stack = torch.stack(test_logits_list, dim=0)
    if combination == 'mean':
        (combined, weights_used) = combine_mean(test_stack)
    else:
        assert val_labels_ref is not None and val_logits_list
        val_stack = torch.stack(val_logits_list, dim=0)
        (_, weights_used) = combine_val_acc_weights(val_stack, val_labels_ref)
        combined = (weights_used.view(-1, 1, 1) * test_stack).sum(dim=0)
    print('\n--- MoE mixture (test) ---', flush=True)
    for (i, name) in enumerate(expert_names):
        print(f'  weight[{name}] = {weights_used[i].item():.4f}', flush=True)
    return combined.argmax(dim=1).tolist()
