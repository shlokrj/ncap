"""Reproducible CPU simulation and visual exports."""

import torch
from .model import NeuralCellularAutomata
from .state import create_seed
from .visualize import render_state


@torch.no_grad()
def export_rollout(model, *, size, steps, seed, output):
    if steps < 0:
        raise ValueError('steps must be nonnegative')
    generator = torch.Generator().manual_seed(seed)
    state = create_seed(channels=model.perceive.channels, height=size, width=size)
    frames = [render_state(state)]
    stride = max(1, steps // 100)
    for step in range(1, steps + 1):
        state = model(state, generator=generator)
        if step % stride == 0 or step == steps:
            frames.append(render_state(state))
    frames[-1].save(output / 'final.png')
    frames[0].save(output / 'growth.gif', save_all=True, append_images=frames[1:],
                   duration=80, loop=0)
    return state


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format')
    config = checkpoint['config']
    model = NeuralCellularAutomata(config['channels'], config['hidden_size'], config['fire_rate'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, config
