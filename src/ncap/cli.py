"""Training and checkpoint simulation commands."""

import argparse
import hashlib
import json
from pathlib import Path
import torch

from .config import TrainConfig
from .simulate import export_rollout, load_checkpoint
from .trainer import train, write_json


def train_main():
    parser = argparse.ArgumentParser(description='Train one RGBA target from a seed (CPU).')
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new run directory; never overwritten')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--steps', type=int, help='override training iterations')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    if args.steps is not None:
        values['iterations'] = args.steps
    if args.seed is not None:
        values['seed'] = args.seed
    metrics = train(args.target, args.output, TrainConfig(**values))
    print(json.dumps(metrics, indent=2))


def simulate_main():
    parser = argparse.ArgumentParser(description='Export a seeded checkpoint rollout (CPU).')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--seed', type=int, default=10000)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error('--steps must be nonnegative')
    model, config = load_checkpoint(args.checkpoint)
    torch.set_num_threads(config['threads'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'status.json', {'status': 'running'})
    try:
        write_json(args.output / 'simulation.json', {
            'checkpoint': str(args.checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'steps': args.steps, 'seed': args.seed, 'config': config,
            'torch': str(torch.__version__), 'device': 'cpu'})
        export_rollout(model, size=config['size'], steps=args.steps, seed=args.seed, output=args.output)
        write_json(args.output / 'status.json', {'status': 'complete'})
    except BaseException as exc:
        write_json(args.output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
