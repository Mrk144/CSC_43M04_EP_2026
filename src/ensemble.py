from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

def _load_model_predictions(entry: Dict[str, Any]) -> Dict[str, Any]:
    val_npz = Path(str(entry['val_npz'])).resolve()
    test_npz = Path(str(entry['test_npz'])).resolve()
    if not val_npz.is_file():
        raise FileNotFoundError(f"Missing val.npz for {entry.get('name', '?')}: {val_npz}")
    if not test_npz.is_file():
        raise FileNotFoundError(f"Missing test.npz for {entry.get('name', '?')}: {test_npz}")
    val = np.load(val_npz, allow_pickle=False)
    test = np.load(test_npz, allow_pickle=False)
    return {'name': str(entry.get('name', val_npz.parent.parent.name)), 'val_names': val['video_names'], 'val_softmax': val['softmax'].astype(np.float32), 'val_labels': val['labels'].astype(np.int64), 'test_names': test['video_names'], 'test_softmax': test['softmax'].astype(np.float32), 'val_accuracy': float(val['val_accuracy']) if 'val_accuracy' in val.files else None}

def _check_alignment(models: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    val_names = models[0]['val_names']
    test_names = models[0]['test_names']
    for m in models[1:]:
        if list(m['val_names']) != list(val_names):
            raise ValueError(f"val video order differs in model {m['name']!r} vs {models[0]['name']!r}")
        if list(m['test_names']) != list(test_names):
            raise ValueError(f"test video order differs in model {m['name']!r} vs {models[0]['name']!r}")
    return (val_names, test_names)

def _normalize_weights(weights: List[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float32)
    s = float(arr.sum())
    if s <= 0:
        raise ValueError('Sum of weights must be > 0.')
    return arr / s

def _resolve_weights(cfg_weights: Any, models: List[Dict[str, Any]]) -> np.ndarray:
    if cfg_weights is None or cfg_weights == 'uniform':
        return np.full(len(models), 1.0 / len(models), dtype=np.float32)
    if cfg_weights == 'auto':
        accs = []
        for m in models:
            if m['val_accuracy'] is None:
                raise ValueError(f"Model {m['name']!r} has no val_accuracy in npz; cannot use weights=auto")
            accs.append(m['val_accuracy'])
        return _normalize_weights(accs)
    if isinstance(cfg_weights, (list, tuple)):
        if len(cfg_weights) != len(models):
            raise ValueError(f'weights length {len(cfg_weights)} != number of models {len(models)}')
        return _normalize_weights([float(x) for x in cfg_weights])
    raise ValueError(f'Unsupported weights spec: {cfg_weights!r}')

def _ensemble_mean(models: List[Dict[str, Any]], weights: np.ndarray, key: str) -> np.ndarray:
    stacked = np.stack([m[key] for m in models], axis=0)
    w = weights.reshape(-1, 1, 1)
    return (stacked * w).sum(axis=0)

def _find_temperature(softmax: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-09
    logits_like = np.log(np.clip(softmax, eps, 1.0))
    best_t = 1.0
    best_nll = float('inf')
    for t in np.linspace(0.5, 3.0, 26):
        scaled = logits_like / float(t)
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t

def _apply_temperature(softmax: np.ndarray, temperature: float) -> np.ndarray:
    eps = 1e-09
    logits_like = np.log(np.clip(softmax, eps, 1.0)) / temperature
    logits_like -= logits_like.max(axis=1, keepdims=True)
    probs = np.exp(logits_like)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)

def _top1_acc(softmax: np.ndarray, labels: np.ndarray) -> float:
    return float((softmax.argmax(axis=1) == labels).mean())

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    ens_cfg = cfg.get('ensemble')
    if ens_cfg is None:
        raise SystemExit("No 'ensemble' section found in config. Run with '+ensemble=default'.")
    print(OmegaConf.to_yaml(ens_cfg))
    entries = list(ens_cfg.models)
    if len(entries) < 2:
        raise SystemExit('Need at least 2 models for ensembling.')
    models = [_load_model_predictions(dict(e)) for e in entries]
    (val_names, test_names) = _check_alignment(models)
    print('Per-model validation accuracy:')
    for m in models:
        acc_str = f"{m['val_accuracy']:.4f}" if m['val_accuracy'] is not None else 'n/a'
        print(f"  - {m['name']}: {acc_str}")
    method = str(ens_cfg.get('method', 'mean_softmax'))
    if method == 'temperature_then_mean':
        for m in models:
            T = _find_temperature(m['val_softmax'], m['val_labels'])
            print(f"  temperature for {m['name']}: T={T:.3f}")
            m['val_softmax'] = _apply_temperature(m['val_softmax'], T)
            m['test_softmax'] = _apply_temperature(m['test_softmax'], T)
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float32)
    elif method == 'weighted_softmax':
        weights = _resolve_weights(ens_cfg.get('weights', 'uniform'), models)
    elif method == 'mean_softmax':
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float32)
    else:
        raise ValueError(f'Unknown ensemble.method: {method}')
    print('Effective weights:', np.round(weights, 4).tolist())
    val_ensemble = _ensemble_mean(models, weights, 'val_softmax')
    val_acc = _top1_acc(val_ensemble, models[0]['val_labels'])
    print(f'\nEnsemble val accuracy ({method}): {val_acc:.4f}')
    best_individual = max((m['val_accuracy'] or _top1_acc(m['val_softmax'], m['val_labels']) for m in models))
    print(f'Best individual val accuracy:       {best_individual:.4f}')
    if val_acc < best_individual:
        print('WARNING: ensemble is below the best individual model. Consider dropping the weakest model or switching method.')
    test_ensemble = _ensemble_mean(models, weights, 'test_softmax')
    test_pred = test_ensemble.argmax(axis=1).astype(np.int64)
    output_csv = Path(str(ens_cfg.output_csv)).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video_name', 'predicted_class'])
        for (name, pred) in zip(test_names, test_pred):
            w.writerow([str(name), int(pred)])
    print(f'Wrote {len(test_pred)} rows to {output_csv}')
if __name__ == '__main__':
    main()
