"""Grid state uses NCHW layout with RGB, alpha, then hidden channels."""

import torch
from torch import Tensor


def validate_state(state: Tensor) -> None:
    if state.ndim != 4 or state.shape[1] < 4 or any(d < 1 for d in state.shape):
        raise ValueError("state must have shape [B, C >= 4, H, W] with positive dimensions")
    if not state.is_floating_point():
        raise ValueError("state must use a floating-point dtype")


def create_seed(batch_size: int = 1, channels: int = 16, height: int = 64,
                width: int = 64, *, device=None, dtype=torch.float32) -> Tensor:
    if min(batch_size, height, width) < 1 or channels < 4:
        raise ValueError("positive batch/grid sizes and at least four channels are required")
    state = torch.zeros(batch_size, channels, height, width, device=device, dtype=dtype)
    validate_state(state)
    state[:, 3:, height // 2, width // 2] = 1
    return state


def state_to_rgba(state: Tensor) -> Tensor:
    """Return the visible channels without clamping or detaching gradients."""
    validate_state(state)
    return state[:, :4]


def rgba_to_state(rgba: Tensor, channels: int = 16) -> Tensor:
    """Append zero hidden channels to batched RGBA tensors."""
    validate_state(rgba)
    if rgba.shape[1] != 4 or channels < 4:
        raise ValueError("input must have four channels; output must have at least four")
    hidden = rgba.new_zeros(rgba.shape[0], channels - 4, *rgba.shape[2:])
    return torch.cat((rgba, hidden), dim=1)
