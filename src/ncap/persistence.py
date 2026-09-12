"""Persistence comparison using every checkpoint from a completed paired study."""

from dataclasses import asdict, replace
import hashlib
from io import BytesIO
import json
from pathlib import Path
from statistics import mean, stdev
import torch

from .config import TrainConfig
from .evaluate import evaluate
from .trainer import write_json


def summarize_persistence(rows):
    """Aggregate evaluation seeds within each trained model, then pair training seeds."""
    steps = sorted({r['step'] for r in rows})
    seeds = sorted({r['training_seed'] for r in rows})
    grouped = {}
    for row in rows:
        key = (row['step'], row['training_seed'], row['variant'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for step in steps:
        for metric in ('loss', 'alpha_iou', 'foreground_cells'):
            pairs = []
            for seed in seeds:
                values = {}
                for variant in ('growth', 'damage'):
                    current = mean(r[metric] for r in grouped[(step, seed, variant)])
                    baseline = mean(r[metric] for r in grouped[(steps[0], seed, variant)])
                    values[variant] = current
                    values[f'{variant}_change'] = current - baseline
                pairs.append(dict(training_seed=seed, **values,
                                  delta=values['damage'] - values['growth']))
            summary.append({'step': step, 'metric': metric, 'baseline_step': steps[0],
                            'training_seed_count': len(seeds),
                            'growth_mean': mean(p['growth'] for p in pairs),
                            'damage_mean': mean(p['damage'] for p in pairs),
                            'growth_change_mean': mean(p['growth_change'] for p in pairs),
                            'damage_change_mean': mean(p['damage_change'] for p in pairs),
                            'paired_delta_mean': mean(p['delta'] for p in pairs),
                            'paired_delta_sd': stdev(p['delta'] for p in pairs) if len(pairs) > 1 else None,
                            'pairs': pairs})
    return summary


def run_persistence(source, output, plan):
    if set(plan) != {'horizons', 'evaluation_seeds'}:
        raise ValueError('plan requires horizons and evaluation_seeds')
    for key in plan:
        values = plan[key]
        if not values or any(type(v) is not int or v < (1 if key == 'horizons' else 0) for v in values) or len(set(values)) != len(values):
            raise ValueError(f'{key} must contain distinct valid integers')
    source, output = Path(source), Path(output)
    if json.loads((source / 'status.json').read_text()).get('status') != 'complete':
        raise ValueError('source study must be complete')
    source_plan_bytes = (source / 'plan.json').read_bytes()
    source_plan = json.loads(source_plan_bytes)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    target_bytes = (source / 'target-source').read_bytes()
    if hashlib.sha256(source_plan_bytes).hexdigest() != source_manifest['plan_sha256'] or hashlib.sha256(target_bytes).hexdigest() != source_manifest['target_sha256']:
        raise ValueError('source study plan or target hash mismatch')
    training_seeds = source_plan['training_seeds']
    if not training_seeds or any(type(s) is not int or s < 0 for s in training_seeds) or len(set(training_seeds)) != len(training_seeds):
        raise ValueError('invalid source training seeds')
    if set(training_seeds) & set(plan['evaluation_seeds']):
        raise ValueError('training and evaluation seeds must be disjoint')
    config = TrainConfig(**source_plan['training'])
    # Snapshot and verify the complete checkpoint set before starting evaluations.
    snapshots = {}
    for seed in training_seeds:
        for variant in ('growth', 'damage'):
            name = f'{variant}-{seed}'
            run = source / name
            if json.loads((run / 'status.json').read_text()).get('status') != 'complete':
                raise ValueError(f'incomplete source run: {name}')
            data = (run / 'checkpoint.pt').read_bytes()
            checkpoint = torch.load(BytesIO(data), weights_only=True, map_location='cpu')
            expected = asdict(replace(config, seed=seed, damage_probability=0 if variant == 'growth' else config.damage_probability))
            if asdict(TrainConfig(**checkpoint['config'])) != expected or checkpoint['iteration'] != expected['iterations']:
                raise ValueError(f'checkpoint settings do not match study: {name}')
            environment = json.loads((run / 'environment.json').read_text())
            if environment['target_sha256'] != source_manifest['target_sha256']:
                raise ValueError(f'checkpoint target does not match study: {name}')
            snapshots[name] = data
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        (output / 'target-source').write_bytes(target_bytes)
        (output / 'source-plan.json').write_bytes(source_plan_bytes)
        write_json(output / 'plan.json', plan)
        write_json(output / 'manifest.json', {
            'source': str(source.resolve()), 'source_plan_sha256': source_manifest['plan_sha256'],
            'target_sha256': source_manifest['target_sha256'],
            'plan_sha256': hashlib.sha256((output / 'plan.json').read_bytes()).hexdigest(),
            'checkpoints': {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()},
            'scope': 'all source models, same target, continuous undamaged trajectories; no stability threshold',
            'replication_unit': 'training seed; evaluation seeds averaged within model'})
        (output / 'checkpoints').mkdir()
        rows = []
        for seed in training_seeds:
            for variant in ('growth', 'damage'):
                name = f'{variant}-{seed}'
                checkpoint_path = output / 'checkpoints' / f'{name}.pt'
                checkpoint_path.write_bytes(snapshots[name])
                print(f'persistence: {name}', flush=True)
                metrics = evaluate(checkpoint_path, output / 'target-source', output / name,
                                   horizons=plan['horizons'], seeds=plan['evaluation_seeds'])
                rows.extend(dict(row, training_seed=seed, variant=variant) for row in metrics)
                write_json(output / 'rows.json', rows)
        summary = summarize_persistence(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete'})
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
