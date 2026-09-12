"""Continuous multi-seed persistence diagnostics on the training target."""

import hashlib
from io import BytesIO
from pathlib import Path
import platform
import torch

from .losses import image_loss
from .simulate import load_checkpoint
from .state import create_seed, load_target
from .trainer import write_json, git_metadata
from .visualize import render_state


@torch.no_grad()
def evaluate(checkpoint, target_path, output, horizons=(64, 96, 192, 384), seeds=(10000, 10001, 10002)):
    horizons, seeds = tuple(horizons), tuple(seeds)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError('horizons must be positive integers')
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    horizons = sorted(set(horizons))
    checkpoint, target_path, output = Path(checkpoint), Path(target_path), Path(output)
    checkpoint_bytes = checkpoint.read_bytes()
    target_bytes = target_path.read_bytes()
    model, config = load_checkpoint(BytesIO(checkpoint_bytes))
    torch.set_num_threads(config['threads'])
    target = load_target(BytesIO(target_bytes), config['size'], config['padding'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        (output / 'target-source').write_bytes(target_bytes)
        write_json(output / 'evaluation.json', {
            'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
            'target_sha256': hashlib.sha256(target_bytes).hexdigest(),
            'config': config, 'horizons': horizons, 'seeds': seeds,
            'torch': str(torch.__version__), 'python': platform.python_version(),
            'device': 'cpu', 'alpha_threshold': 0.1, **git_metadata(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))},
            'scope': 'continuous rollouts on supplied target; diagnostic, not held-out evaluation'})
        results = []
        for seed in seeds:
            state = create_seed(channels=config['channels'], height=config['size'], width=config['size'])
            generator = torch.Generator().manual_seed(seed)
            previous = 0
            for horizon in horizons:
                state = model.rollout(state, horizon - previous, generator=generator)
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f'nonfinite state at seed {seed}, step {horizon}')
                foreground = state[:, 3:4] > 0.1
                expected = target[:, 3:4] > 0.1
                intersection = (foreground & expected).sum().item()
                union = (foreground | expected).sum().item()
                results.append({'seed': seed, 'step': horizon, 'loss': image_loss(state, target).item(),
                                'alpha_iou': intersection / union if union else 1.0,
                                'foreground_cells': foreground.sum().item(),
                                'state_abs_max': state.abs().max().item()})
                render_state(state).save(output / f'seed-{seed}-step-{horizon}.png')
                previous = horizon
                write_json(output / 'metrics.json', results)
        write_json(output / 'metrics.json', results)
        write_json(output / 'status.json', {'status': 'complete'})
        return results
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
