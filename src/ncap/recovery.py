"""Paired damaged/undamaged recovery diagnostics with matching stochastic updates."""

import hashlib
from io import BytesIO
from pathlib import Path
import platform
import torch

from .damage import apply_damage, GEOMETRIES
from .losses import image_loss
from .simulate import load_checkpoint
from .state import create_seed, load_target
from .trainer import write_json, git_metadata
from .visualize import render_state


@torch.no_grad()
def evaluate_recovery(checkpoint, target_path, output, *, grow_steps=96, recovery_steps=96,
                      fractions=(0.1, 0.25, 0.5), seeds=(20000, 20001, 20002), geometry='dropout'):
    if geometry not in GEOMETRIES:
        raise ValueError('unknown damage geometry')
    fractions, seeds = tuple(fractions), tuple(seeds)
    if any(type(v) is not int or v < 1 for v in (grow_steps, recovery_steps)):
        raise ValueError('growth and recovery steps must be positive integers')
    if not fractions or any(not 0 <= f <= 1 for f in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError('fractions must be unique values in [0, 1]')
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be unique nonnegative integers')
    checkpoint_bytes = Path(checkpoint).read_bytes()
    target_bytes = Path(target_path).read_bytes()
    model, config = load_checkpoint(BytesIO(checkpoint_bytes))
    torch.set_num_threads(config['threads'])
    target = load_target(BytesIO(target_bytes), config['size'], config['padding'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        (output / 'target-source').write_bytes(target_bytes)
        write_json(output / 'evaluation.json', {
            'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
            'target_sha256': hashlib.sha256(target_bytes).hexdigest(),
            'config': config, 'grow_steps': grow_steps, 'recovery_steps': recovery_steps,
            'fractions': fractions, 'seeds': seeds, 'geometry': geometry, 'torch': str(torch.__version__),
            'python': platform.python_version(), 'device': 'cpu', **git_metadata(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))},
            'damage': 'exact rounded fraction of all grid cells; all channels removed',
            'scope': 'same-target paired diagnostics; no thresholded success or recovery-time claim'})
        rows = []
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            initial = create_seed(channels=config['channels'], height=config['size'], width=config['size'])
            grown = model.rollout(initial, grow_steps, generator=generator)
            update_rng = generator.get_state()
            control = model.rollout(grown, recovery_steps, generator=generator)
            if not torch.isfinite(grown).all() or not torch.isfinite(control).all():
                raise FloatingPointError('nonfinite growth or control state')
            render_state(grown).save(output / f'seed-{seed}-grown.png')
            render_state(control).save(output / f'seed-{seed}-control.png')
            for index, fraction in enumerate(fractions):
                damaged = apply_damage(grown, fraction, geometry, generator=torch.Generator().manual_seed(seed))
                generator.set_state(update_rng)
                recovered = model.rollout(damaged, recovery_steps, generator=generator)
                if not torch.isfinite(recovered).all():
                    raise FloatingPointError('nonfinite recovery state')
                original_foreground = grown[:, 3:4] > 0.1
                removed = original_foreground & (damaged[:, 3:4] <= 0.1)
                count = original_foreground.sum().item()
                rows.append({'seed': seed, 'fraction': fraction, 'geometry': geometry,
                             'foreground_removed_fraction': removed.sum().item() / count if count else None,
                             'grown_loss': image_loss(grown, target).item(),
                             'damaged_loss': image_loss(damaged, target).item(),
                             'recovered_loss': image_loss(recovered, target).item(),
                             'control_loss': image_loss(control, target).item()})
                render_state(damaged).save(output / f'seed-{seed}-damage-{index}.png')
                render_state(recovered).save(output / f'seed-{seed}-recovery-{index}.png')
                write_json(output / 'metrics.json', rows)
        write_json(output / 'status.json', {'status': 'complete'})
        return rows
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
