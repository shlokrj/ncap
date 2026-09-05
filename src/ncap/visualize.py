"""Display premultiplied states on white without changing training tensors."""

import torch
from PIL import Image
from .state import state_to_rgba


def render_state(state) -> Image.Image:
    rgba = state_to_rgba(state).detach().cpu()[0].clamp(0, 1)
    rgb = (rgba[:3] + 1 - rgba[3:4]).clamp(0, 1)
    pixels = (rgb.permute(1, 2, 0) * 255).round().to(dtype=torch.uint8)
    return Image.frombytes('RGB', (state.shape[3], state.shape[2]), bytes(pixels.flatten().tolist()))
