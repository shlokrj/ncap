"""Seed or state-pool training with immutable run folders and auditable artifacts."""

import csv
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import time

from PIL import __version__ as pillow_version
import torch

from .damage import cell_dropout
from .config import TrainConfig
from .losses import image_loss
from .pool import StatePool
from .model import NeuralCellularAutomata
from .simulate import export_rollout
from .state import create_seed, load_target
from .visualize import render_state


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def git_metadata():
    try:
        root = Path(__file__).resolve().parents[2]
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                               capture_output=True, text=True)
        return {
            'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
            'git_dirty': bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except OSError:
        return {'git_revision': None, 'git_dirty': None}


def train(target_path, output, config: TrainConfig):
    target_path, output = Path(target_path), Path(output)
    # Snapshot first: the exact bytes used for target loading are retained in the run.
    target_bytes = target_path.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        (output / 'target-source').write_bytes(target_bytes)
        target = load_target(output / 'target-source', config.size, config.padding)
        if not torch.any(target[:, 3] > 0):
            raise ValueError('target must contain visible pixels')
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        rng = random.Random(config.seed)
        generator = torch.Generator().manual_seed(config.seed)
        model = NeuralCellularAutomata(config.channels, config.hidden_size, config.fire_rate, config.state_limit)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        write_json(output / 'config.json', asdict(config))
        write_json(output / 'environment.json', {
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'pillow': pillow_version, 'platform': platform.platform(), 'device': 'cpu',
            **git_metadata(),
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(Path(__file__).parent.glob('*.py'))},
            'target_path': str(target_path.resolve()),
            'target_sha256': hashlib.sha256(target_bytes).hexdigest(),
        })
        render_state(target).save(output / 'target.png')
        seed = create_seed(config.batch_size, config.channels, config.size, config.size)
        pool = StatePool(seed[:1], config.pool_size) if config.pool_size else None
        pool_generator = torch.Generator().manual_seed(config.seed)
        damage_generator = torch.Generator().manual_seed(config.seed)
        started = time.monotonic()
        with (output / 'loss.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['iteration', 'rollout_steps', 'loss'])
            for iteration in range(1, config.iterations + 1):
                steps = rng.randint(config.min_steps, config.max_steps)
                optimizer.zero_grad(set_to_none=True)
                if pool is not None:
                    indices, batch = pool.sample(config.batch_size, target, generator=pool_generator)
                else:
                    batch = seed
                if config.damage_probability:
                    # Keep freshly injected seeds intact; damage only previously grown states.
                    eligible = (batch != seed[:1]).flatten(1).any(dim=1)
                    selected = torch.rand(config.batch_size, generator=damage_generator) < config.damage_probability
                    selected &= eligible
                    if selected.any():
                        batch[selected] = cell_dropout(batch[selected], config.damage_fraction,
                                                       generator=damage_generator)
                state = model.rollout(batch, steps, generator=generator)
                loss = image_loss(state, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if pool is not None:
                    pool.commit(indices, state)
                writer.writerow([iteration, steps, loss.item()])
                stream.flush()
                if iteration == 1 or iteration % 100 == 0 or iteration == config.iterations:
                    print(f'iteration {iteration}/{config.iterations}: loss={loss.item():.6f}', flush=True)
        torch.save({'format_version': 1, 'config': asdict(config), 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'iteration': config.iterations,
                    'pool': pool.states if pool is not None else None,
                    'pool_generator_state': pool_generator.get_state(),
                    'damage_generator_state': damage_generator.get_state(),
                    'generator_state': generator.get_state(), 'python_rng_state': rng.getstate()},
                   output / 'checkpoint.pt')
        model.eval()
        final = export_rollout(model, size=config.size, steps=config.eval_steps,
                               seed=config.eval_seed, output=output)
        metrics = {'final_loss': image_loss(final, target).item(),
                   'seed_loss': image_loss(seed[:1], target).item(),
                   'seconds': time.monotonic() - started,
                   'evaluation': 'single seeded rollout; not a stability or generalization evaluation'}
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'status.json', {'status': 'complete'})
        return metrics
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
