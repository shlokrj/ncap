"""A stochastic residual rule shared by every cell in the grid."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .perception import SobelPerception
from .state import validate_state


class NeuralCellularAutomata(nn.Module):
    def __init__(self, channels: int = 16, hidden_size: int = 128,
                 fire_rate: float = 0.5):
        super().__init__()
        if hidden_size < 1 or not 0 <= fire_rate <= 1:
            raise ValueError("hidden_size must be positive and fire_rate must be in [0, 1]")
        self.fire_rate = fire_rate
        self.perceive = SobelPerception(channels)
        self.update = nn.Sequential(nn.Conv2d(channels * 3, hidden_size, 1),
                                    nn.ReLU(), nn.Conv2d(hidden_size, channels, 1))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    @staticmethod
    def alive_mask(state: Tensor) -> Tensor:
        validate_state(state)
        return F.max_pool2d(state[:, 3:4], kernel_size=3, stride=1, padding=1) > 0.1

    def forward(self, state: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        alive_before = self.alive_mask(state)
        delta = self.update(self.perceive(state))
        fired = torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device,
                           generator=generator) < self.fire_rate
        updated = state + delta * fired
        return updated * (alive_before & self.alive_mask(updated))

    def rollout(self, state: Tensor, steps: int, *,
                generator: torch.Generator | None = None) -> Tensor:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        validate_state(state)
        if state.shape[1] != self.perceive.channels:
            raise ValueError("state channels do not match model channels")
        for _ in range(steps):
            state = self(state, generator=generator)
        return state
