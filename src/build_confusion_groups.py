#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hydra
import yaml
from omegaconf import DictConfig
from confusion_groups import build_confusion_groups, load_confusion_counts_csv, name_group, parse_class_folder_names

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
    b = cfg.get('build')
    if b is None:
        raise SystemExit('Use: python src/build_confusion_groups.py +build_confusion_groups=default')
    cm_path = Path(str(b.confusion_csv)).resolve()
    (cm, support) = load_confusion_counts_csv(cm_path)
    groups = build_confusion_groups(cm, support, min_rate=float(b.get('min_rate', 0.1)), min_pair_errors=int(b.get('min_pair_errors', 12)), min_group_size=int(b.get('min_group_size', 2)))
    val_root = Path(str(b.get('val_dir', cfg.dataset.val_dir))).resolve()
    class_names = parse_class_folder_names(val_root) if val_root.is_dir() else {}
    named = []
    for g in groups:
        ids = g['class_ids']
        named.append({'name': name_group(ids, class_names), 'class_ids': ids, 'top_pairs': g.get('top_pairs', [])})
    out_path = Path(str(b.get('output_yaml', 'src/configs/confusion_groups/from_tsm_resnet.yaml'))).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'source_confusion': str(cm_path), 'confusion_catalog': {'groups': named}}
    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    print(f'Wrote {len(named)} groups to {out_path}')
    for g in named:
        ids = g['class_ids']
        labels = [class_names.get(c, str(c)) for c in ids]
        print(f"  - {g['name']}: {ids}")
        for lab in labels:
            print(f'      {lab}')
if __name__ == '__main__':
    main()
