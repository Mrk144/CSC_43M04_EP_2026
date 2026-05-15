"""Temporal interpolation utilities embedded into model forward passes.

Why this lives inside the model and not the dataset: at submission time the
contest only loads our saved ``.pt`` (state dict + architecture). The test
loader serves the raw frames on disk (T = 4 in this challenge). If we want the
backbone to operate at T = 9 (Track A) or T = 16 (Track B), the upsampling
must happen *inside the model*, so the ``.pt`` is fully self-contained.

The interpolation is deterministic and matches the one previously done in the
dataset: ``T`` evenly spaced positions on ``[0, T_in - 1]`` blended linearly
between the two nearest real frames. Pure tensor ops, runs on GPU, batched and
differentiable.
"""

from __future__ import annotations

import torch


def temporal_interpolate(video: torch.Tensor, target_T: int) -> torch.Tensor:
    """Linearly resample the temporal axis of a video tensor.

    Args:
        video: shape ``(B, T_in, C, H, W)``.
        target_T: number of frames after interpolation. Must be >= 1.

    Returns:
        Tensor of shape ``(B, target_T, C, H, W)`` on the same device/dtype.

    Behavior:
        - ``target_T == T_in``: returns the input unchanged.
        - ``T_in == 1``: replicates the single frame ``target_T`` times.
        - Otherwise: position ``i`` of the output is
          ``(1 - alpha) * frame_lo + alpha * frame_hi`` with
          ``pos = i * (T_in - 1) / (target_T - 1)``, ``lo = floor(pos)``,
          ``hi = ceil(pos)``, ``alpha = pos - lo``.
    """
    if video.dim() != 5:
        raise ValueError(
            f"Expected (B, T, C, H, W) tensor, got shape {tuple(video.shape)}"
        )
    if target_T <= 0:
        raise ValueError(f"target_T must be positive, got {target_T}")

    _B, T_in, _C, _H, _W = video.shape

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
