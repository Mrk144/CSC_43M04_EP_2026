#!/usr/bin/env python3
from __future__ import annotations
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from checkpoint_utils import aug_cfg_for_expert, expert_eval_settings, load_model_from_checkpoint
from dataset.video_dataset import VideoFrameDataset
from moe_core import run_moe_submission
from utils import build_transforms, forward_logits_with_tta, get_augmentation_cfg, set_seed

def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'video_name' not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row['video_name'].strip() for row in reader]

def _index_video_folders(test_root: Path) -> Dict[str, Path]:
    test_root = test_root.resolve()
    index: Dict[str, Path] = {}
    for (dirpath, dirs, _files) in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith('video_'):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(f'Duplicate video folder name {name!r}: {index[name]} and {p}')
            index[name] = p
            dirs.remove(name)
    return index

def resolve_video_dirs(test_root: Path, video_names: List[str]) -> List[Path]:
    index = _index_video_folders(test_root)
    out: List[Path] = []
    missing: List[str] = []
    for name in video_names:
        p = index.get(name)
        if p is None:
            missing.append(name)
        else:
            out.append(p)
    if missing:
        sample = ', '.join((repr(m) for m in missing[:5]))
        extra = f' (+{len(missing) - 5} more)' if len(missing) > 5 else ''
        raise FileNotFoundError(f'{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}')
    return out

def discover_all_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    index = _index_video_folders(test_root)
    video_names = sorted(index.keys())
    video_dirs = [index[name] for name in video_names]
    return (video_names, video_dirs)

def resolve_test_videos(cfg: DictConfig) -> Tuple[List[str], List[Path], Path]:
    test_root = Path(cfg.dataset.test_dir).resolve()
    manifest_cfg = cfg.dataset.get('test_manifest')
    print(f'Indexing video folders under: {test_root}', flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f'Reading manifest: {manifest_path}', flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
        print(f'Resolved {len(video_dirs)} video folders from manifest for inference.', flush=True)
    else:
        print('No dataset.test_manifest provided; using all video_* folders found in test_dir.', flush=True)
        (video_names, video_dirs) = discover_all_test_videos(test_root)
        print(f'Discovered {len(video_dirs)} video folders (sorted by video name).', flush=True)
    return (video_names, video_dirs, test_root)

def _write_submission_csv(output_path: Path, video_names: List[str], predictions: List[int], header: Tuple[str, ...]=('video_name', 'predicted_class')) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Writing submission CSV: {output_path}', flush=True)
    with output_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for (name, pred) in zip(video_names, predictions):
            w.writerow([name, pred])
    print(f'  -> {len(predictions)} rows', flush=True)

@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device, total_videos: int, model_label: str='', aug_cfg: DictConfig | None=None) -> List[int]:
    model.eval()
    aug_cfg = aug_cfg or OmegaConf.create({})
    preds: List[int] = []
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    processed = 0
    prefix = f'[{model_label}] ' if model_label else ''
    use_autocast = device.type == 'cuda'
    for (batch_idx, (video_batch, _labels)) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        logits = forward_logits_with_tta(model, video_batch, aug_cfg, use_autocast=use_autocast)
        batch_pred = logits.argmax(dim=1).cpu().tolist()
        preds.extend((int(p) for p in batch_pred))
        bs = video_batch.size(0)
        processed += bs
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(f'{prefix}Inference batch {batch_idx}/{n_batches} ({processed}/{total_videos} clips)', flush=True)
    return preds

def _predict_one_checkpoint(checkpoint_path: Path, video_names: List[str], video_dirs: List[Path], test_root: Path, cfg: DictConfig, device: torch.device, model_label: str) -> List[int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    print(f'Loading checkpoint: {checkpoint_path}', flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = load_model_from_checkpoint(ckpt, device)
    settings = expert_eval_settings(ckpt, cfg)
    eval_transform = build_transforms(is_training=False, image_size=settings['image_size'])
    aug_cfg = aug_cfg_for_expert(cfg, settings['model_name'])
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    dataset = VideoFrameDataset(root_dir=test_root, num_frames=settings['num_frames'], transform=eval_transform, sample_list=sample_list)
    batch_size = settings['batch_size']
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg.training.num_workers), pin_memory=device.type == 'cuda')
    print(f'[{model_label}] Starting inference: {len(dataset)} clips, batch_size={batch_size}, {len(loader)} batches', flush=True)
    predictions = run_inference(model, loader, device, total_videos=len(dataset), model_label=model_label, aug_cfg=aug_cfg)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if len(predictions) != len(video_names):
        raise RuntimeError(f'[{model_label}] Prediction count {len(predictions)} != {len(video_names)}')
    return predictions

def _majority_vote(rows: List[List[int]]) -> List[int]:
    n = len(rows[0])
    out: List[int] = []
    for i in range(n):
        votes = [rows[k][i] for k in range(len(rows))]
        counts = Counter(votes)
        best_count = max(counts.values())
        winners = [c for (c, cnt) in counts.items() if cnt == best_count]
        out.append(min(winners))
    return out

def _run_multi_model_submission(cfg: DictConfig, video_names: List[str], video_dirs: List[Path], test_root: Path, device: torch.device) -> None:
    sub_cfg = cfg.submission
    models_cfg = list(sub_cfg.models)
    if len(models_cfg) < 1:
        raise SystemExit('submission.models must list at least one checkpoint.')
    all_preds: Dict[str, List[int]] = {}
    for entry in models_cfg:
        entry_d = OmegaConf.to_container(entry, resolve=True)
        assert isinstance(entry_d, dict)
        name = str(entry_d.get('name', Path(str(entry_d['checkpoint'])).parent.name))
        ckpt_path = Path(str(entry_d['checkpoint'])).resolve()
        all_preds[name] = _predict_one_checkpoint(ckpt_path, video_names, video_dirs, test_root, cfg, device, name)
    names = list(all_preds.keys())
    pred_matrix = [all_preds[n] for n in names]
    if bool(sub_cfg.get('write_per_model', False)):
        per_model_dir = sub_cfg.get('per_model_dir')
        if not per_model_dir:
            raise ValueError('submission.write_per_model=true requires submission.per_model_dir.')
        out_dir = Path(str(per_model_dir)).resolve()
        for name in names:
            _write_submission_csv(out_dir / f'submission_{name}.csv', video_names, all_preds[name])
    ensemble_mode = str(sub_cfg.get('ensemble_column', 'majority'))
    if ensemble_mode == 'majority':
        ensemble_preds = _majority_vote(pred_matrix)
    elif ensemble_mode == 'first':
        ensemble_preds = pred_matrix[0]
    elif ensemble_mode == 'none':
        raise ValueError('submission.ensemble_column=none requires output_format=wide (no single predicted_class to write).')
    else:
        raise ValueError(f'Unknown submission.ensemble_column={ensemble_mode!r}; use majority | first')
    combined_path = Path(str(sub_cfg.get('output_csv', cfg.dataset.submission_output))).resolve()
    output_format = str(sub_cfg.get('output_format', 'simple'))
    if output_format == 'simple':
        _write_submission_csv(combined_path, video_names, ensemble_preds)
        print(f'\n=== Soumission ensemble ({ensemble_mode}, {len(names)} modèles) ===\nFichier à déposer : {combined_path}\n(2 colonnes : video_name, predicted_class)\n', flush=True)
    elif output_format == 'wide':
        header = ['video_name'] + [f'pred_{n}' for n in names] + ['predicted_class']
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Writing wide CSV: {combined_path}', flush=True)
        with combined_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(header)
            for (i, name) in enumerate(video_names):
                row: List[Any] = [name] + [all_preds[n][i] for n in names] + [ensemble_preds[i]]
                w.writerow(row)
        print(f'Done. Wide file: {combined_path} ({len(video_names)} rows)', flush=True)
    else:
        raise ValueError(f'Unknown submission.output_format={output_format!r}; use simple | wide')

def _run_moe_submission(cfg: DictConfig, video_names: List[str], video_dirs: List[Path], test_root: Path, device: torch.device) -> None:
    predictions = run_moe_submission(cfg, video_dirs, test_root, device)
    moe_cfg = cfg.moe
    out = Path(str(moe_cfg.get('output_csv', cfg.dataset.submission_output))).resolve()
    _write_submission_csv(out, video_names, predictions)
    print(f"\n=== Soumission MoE ({moe_cfg.get('combination', 'mean')}) ===\nFichier : {out}\n", flush=True)

def _run_single_model_submission(cfg: DictConfig, video_names: List[str], video_dirs: List[Path], test_root: Path, device: torch.device) -> None:
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    predictions = _predict_one_checkpoint(checkpoint_path, video_names, video_dirs, test_root, cfg, device, model_label='model')
    output_path = Path(cfg.dataset.submission_output).resolve()
    _write_submission_csv(output_path, video_names, predictions)
    print(f'Done. Wrote {len(predictions)} rows to {output_path}', flush=True)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))
    device_str = cfg.training.device
    if device_str == 'cuda' and (not torch.cuda.is_available()):
        print('CUDA not available; using CPU.')
        device_str = 'cpu'
    device = torch.device(device_str)
    (video_names, video_dirs, test_root) = resolve_test_videos(cfg)
    if cfg.get('moe') is not None and cfg.moe.get('experts'):
        _run_moe_submission(cfg, video_names, video_dirs, test_root, device)
    elif cfg.get('submission') is not None and cfg.submission.get('models'):
        _run_multi_model_submission(cfg, video_names, video_dirs, test_root, device)
    else:
        _run_single_model_submission(cfg, video_names, video_dirs, test_root, device)
if __name__ == '__main__':
    main()
