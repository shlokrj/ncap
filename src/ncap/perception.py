"""Fixed, channel-wise identity and normalized Sobel perception."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .state import validate_state


class SobelPerception(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        if channels < 4:
            raise ValueError("at least four channels are required")
        self.channels = channels
        identity = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        dx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        kernels = torch.stack((identity, dx, dx.T))[:, None]
        self.register_buffer("kernels", kernels.repeat(channels, 1, 1, 1))

    def forward(self, state: Tensor) -> Tensor:
        """Return [identity, dx, dy] per channel, with zero boundary padding."""
        validate_state(state)
        if state.shape[1] != self.channels:
            raise ValueError("state channels do not match perception channels")
        return F.conv2d(state, self.kernels, padding=1, groups=self.channels)
