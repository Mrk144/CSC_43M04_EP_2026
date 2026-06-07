from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

def _list_frame_paths(video_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for extension in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)

def _parse_class_index(class_dir_name: str) -> Optional[int]:
    match = re.match('^(\\d+)_', class_dir_name)
    if match is None:
        return None
    return int(match.group(1))

def collect_video_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f'Dataset root not found: {root_dir}')
    samples: List[Tuple[Path, int]] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    fallback_index = {p.name: i for (i, p) in enumerate(class_dirs)}
    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue
            samples.append((video_dir, class_index))
    if len(samples) == 0:
        raise RuntimeError(f'No video folders with frames under {root_dir}')
    return samples

def _pick_frame_positions(num_available: int, num_frames: int) -> List[float]:
    if num_available <= 0:
        raise ValueError('Video has no frames.')
    if num_frames <= 0:
        raise ValueError('num_frames must be positive.')
    if num_available == 1:
        return [0.0] * num_frames
    positions = torch.linspace(0.0, num_available - 1, steps=num_frames)
    return [float(x) for x in positions]

def _load_frame_uint8(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb_image = image.convert('RGB')
        return TF.pil_to_tensor(rgb_image)

def _load_frame_at_position(frame_paths: List[Path], pos: float, cache: dict) -> torch.Tensor:
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    lo = max(0, min(lo, len(frame_paths) - 1))
    hi = max(0, min(hi, len(frame_paths) - 1))
    if lo not in cache:
        cache[lo] = _load_frame_uint8(frame_paths[lo])
    if lo == hi:
        return cache[lo]
    if hi not in cache:
        cache[hi] = _load_frame_uint8(frame_paths[hi])
    alpha = pos - lo
    blended = (1.0 - alpha) * cache[lo].to(torch.float32) + alpha * cache[hi].to(torch.float32)
    return blended.clamp_(0.0, 255.0).to(torch.uint8)

class VideoFrameDataset(Dataset):

    def __init__(self, root_dir: str | Path, num_frames: int, transform: Callable[[torch.Tensor], torch.Tensor], sample_list: Optional[List[Tuple[Path, int]]]=None, hflip_prob: float=0.0) -> None:
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.hflip_prob = float(hflip_prob)
        if sample_list is None:
            self.samples = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (video_dir, label) = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        positions = _pick_frame_positions(len(frame_paths), self.num_frames)
        cache: dict = {}
        tensors: List[torch.Tensor] = [_load_frame_at_position(frame_paths, pos, cache) for pos in positions]
        video_tensor = torch.stack(tensors, dim=0)
        if self.hflip_prob > 0.0:
            from utils import apply_horizontal_flip_video
            (video_tensor, label) = apply_horizontal_flip_video(video_tensor, label, prob=self.hflip_prob)
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return (video_tensor, label_tensor)
