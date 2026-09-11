"""Visible-channel reconstruction objective in premultiplied RGBA space."""

from torch import Tensor
from .state import validate_state


def image_loss(state: Tensor, target: Tensor) -> Tensor:
    validate_state(state)
    validate_state(target)
    if target.shape[1] != 4 or target.shape[2:] != state.shape[2:]:
        raise ValueError('target must be RGBA with the same grid dimensions')
    if target.shape[0] not in (1, state.shape[0]):
        raise ValueError('target batch must be one or match the state batch')
    return (state[:, :4] - target).square().mean()


def excess_state_loss(state: Tensor, threshold: float = 2.0) -> Tensor:
    """Mean squared excess magnitude across every channel at the rollout endpoint."""
    import math
    validate_state(state)
    if not math.isfinite(threshold) or threshold < 1:
        raise ValueError('threshold must be finite and at least one')
    return (state.abs() - threshold).clamp_min(0).square().mean()
