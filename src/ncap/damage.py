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


GEOMETRIES = ('dropout', 'center', 'edge', 'horizontal')


def apply_damage(state, fraction, geometry, *, generator):
    """Remove an exact grid-cell count using a fixed spatial ordering or random dropout."""
    validate_state(state)
    if geometry not in GEOMETRIES or not 0 <= fraction <= 1:
        raise ValueError('unknown geometry or fraction outside [0, 1]')
    if geometry == 'dropout':
        return cell_dropout(state, fraction, generator=generator)
    height, width = state.shape[2:]
    y, x = torch.meshgrid(torch.arange(height, device=state.device),
                          torch.arange(width, device=state.device), indexing='ij')
    if geometry == 'center':
        distance = (y - (height - 1) / 2).square() + (x - (width - 1) / 2).square()
    elif geometry == 'edge':
        distance = x  # Left-to-right cut, with row-major ties.
    else:
        distance = (y - (height - 1) / 2).abs()
    indices = torch.argsort(distance.flatten(), stable=True)
    keep = torch.ones(height * width, device=state.device, dtype=state.dtype)
    keep[indices[:round(fraction * height * width)]] = 0
    return state * keep.reshape(height, width)
