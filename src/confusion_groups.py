"""Build class groups from a confusion-matrix CSV (symmetric confusion graph)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_confusion_counts_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``confusion_counts_*.csv`` -> (cm [C,C], support [C])."""
    path = path.resolve()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        pred_ids = [int(x) for x in header[1:]]
        n = len(pred_ids)
        cm = np.zeros((n, n), dtype=np.int64)
        for row in reader:
            true_id = int(row[0])
            for j, val in enumerate(row[1:]):
                cm[true_id, pred_ids[j]] = int(val)
    support = cm.sum(axis=1)
    return cm, support


def build_confusion_groups(
    cm: np.ndarray,
    support: np.ndarray,
    min_rate: float = 0.10,
    min_pair_errors: int = 12,
    min_group_size: int = 2,
) -> List[Dict[str, object]]:
    """Connected components on edges where A<->B confuse often.

    Undirected edge (i, j), i != j, when::
        cm[i,j]/support[i] >= min_rate OR cm[j,i]/support[j] >= min_rate
    and ``cm[i,j] + cm[j,i] >= min_pair_errors``.
    """
    n = cm.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if support[i] <= 0:
            continue
        for j in range(i + 1, n):
            if support[j] <= 0:
                continue
            err_ij = int(cm[i, j])
            err_ji = int(cm[j, i])
            rate_ij = err_ij / float(support[i])
            rate_ji = err_ji / float(support[j])
            if err_ij + err_ji < min_pair_errors:
                continue
            if rate_ij >= min_rate or rate_ji >= min_rate:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for c in range(n):
        if support[c] <= 0:
            continue
        root = find(c)
        clusters.setdefault(root, []).append(c)

    groups: List[Dict[str, object]] = []
    for class_ids in sorted(clusters.values(), key=lambda xs: (len(xs), min(xs))):
        if len(class_ids) < min_group_size:
            continue
        pairs: List[Tuple[int, int, int, float]] = []
        for i in class_ids:
            for j in class_ids:
                if i == j:
                    continue
                err = int(cm[i, j])
                if err > 0:
                    pairs.append((i, j, err, err / max(float(support[i]), 1.0)))
        pairs.sort(key=lambda t: -t[2])
        groups.append(
            {
                "class_ids": sorted(class_ids),
                "top_pairs": [
                    {"true": i, "pred": j, "count": c, "rate": round(r, 4)}
                    for i, j, c, r in pairs[:8]
                ],
            }
        )
    return groups


def name_group(class_ids: Sequence[int], class_names: Dict[int, str] | None = None) -> str:
    if class_names:
        short = [class_names.get(c, str(c)).split("_")[0][:12] for c in class_ids[:3]]
        return "grp_" + "_".join(short) + (f"_plus{len(class_ids)-3}" if len(class_ids) > 3 else "")
    return "grp_" + "_".join(str(c) for c in class_ids)


def parse_class_folder_names(val_root: Path) -> Dict[int, str]:
    import re

    out: Dict[int, str] = {}
    for p in val_root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(r"^(\d+)_", p.name)
        if m:
            out[int(m.group(1))] = p.name
    return out
