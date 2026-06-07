from __future__ import annotations
import torch

def temporal_interpolate(video: torch.Tensor, target_T: int) -> torch.Tensor:
    if video.dim() != 5:
        raise ValueError(f'Expected (B, T, C, H, W) tensor, got shape {tuple(video.shape)}')
    if target_T <= 0:
        raise ValueError(f'target_T must be positive, got {target_T}')
    (_B, T_in, _C, _H, _W) = video.shape
    if T_in == target_T:
        return video
    if T_in == 1:
        return video.expand(-1, target_T, -1, -1, -1).clone()
    device = video.device
    dtype = video.dtype
    positions = torch.linspace(0.0, T_in - 1, steps=target_T, device=device)
    lo = positions.floor().clamp_(min=0, max=T_in - 1).long()
    hi = positions.ceil().clamp_(min=0, max=T_in - 1).long()
    alpha = (positions - lo.to(positions.dtype)).to(dtype)
    lo_frames = video.index_select(dim=1, index=lo)
    hi_frames = video.index_select(dim=1, index=hi)
    alpha = alpha.view(1, target_T, 1, 1, 1)
    return (1.0 - alpha) * lo_frames + alpha * hi_frames
