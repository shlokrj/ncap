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


def evaluate_main():
    from .evaluate import evaluate
    parser = argparse.ArgumentParser(description='Measure continuous multi-seed growth and persistence.')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--horizons', nargs='+', type=int, default=[64, 96, 192, 384])
    parser.add_argument('--seeds', nargs='+', type=int, default=[10000, 10001, 10002])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.target, args.output, args.horizons, args.seeds), indent=2))


def recovery_main():
    from .recovery import evaluate_recovery
    from .damage import GEOMETRIES
    parser = argparse.ArgumentParser(description='Compare damaged recovery with matching undamaged rollouts.')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--geometry', choices=GEOMETRIES, default='dropout')
    parser.add_argument('--grow-steps', type=int, default=96)
    parser.add_argument('--recovery-steps', type=int, default=96)
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20000, 20001, 20002])
    args = parser.parse_args()
    print(json.dumps(evaluate_recovery(args.checkpoint, args.target, args.output,
                                      grow_steps=args.grow_steps, recovery_steps=args.recovery_steps,
                                      fractions=args.fractions, seeds=args.seeds, geometry=args.geometry), indent=2))


def study_main():
    from .experiments import run_study
    parser = argparse.ArgumentParser(description='Run a fixed paired multi-seed damage-training study.')
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_study(args.target, args.output, json.loads(args.plan.read_text())), indent=2))


def persistence_main():
    from .persistence import run_persistence
    parser = argparse.ArgumentParser(description='Compare persistence of every model in a completed study.')
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_persistence(args.study, args.output, json.loads(args.plan.read_text())), indent=2))
