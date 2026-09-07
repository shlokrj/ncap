"""Non-mutating damage removes every channel at selected grid cells."""

import math
import torch
from .state import validate_state


def circle_damage(state, center, radius):
    validate_state(state)
    if len(center) != 2 or not all(math.isfinite(v) for v in center) or not math.isfinite(radius) or radius < 0:
        raise ValueError('center and nonnegative radius must be finite')
    y, x = torch.meshgrid(torch.arange(state.shape[2], device=state.device),
                          torch.arange(state.shape[3], device=state.device), indexing='ij')
    keep = (y - center[0]).square() + (x - center[1]).square() > radius ** 2
    return state * keep


def rectangle_damage(state, top, left, height, width):
    validate_state(state)
    if any(type(v) is not int or v < 0 for v in (top, left, height, width)):
        raise ValueError('rectangle coordinates and sizes must be nonnegative integers')
    if top + height > state.shape[2] or left + width > state.shape[3]:
        raise ValueError('rectangle must fit within the grid')
    keep = torch.ones_like(state[:, :1])
    keep[:, :, top:top + height, left:left + width] = 0
    return state * keep


def cell_dropout(state, fraction, *, generator):
    """Remove exactly round(fraction * H * W) grid cells per sample, including background."""
    validate_state(state)
    if not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    count = state.shape[2] * state.shape[3]
    keep = torch.ones(state.shape[0], count, device=state.device, dtype=state.dtype)
    for row in keep:
        indices = torch.randperm(count, device=state.device, generator=generator)
        row[indices[:round(fraction * count)]] = 0
    return state * keep.reshape(state.shape[0], 1, *state.shape[2:])
