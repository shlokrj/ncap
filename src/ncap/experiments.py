"""Fixed paired damage-training comparison; training seeds are the replication unit."""

from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean, stdev

from .config import TrainConfig
from .damage import GEOMETRIES
from .recovery import evaluate_recovery
from .trainer import train, write_json


def summarize(rows):
    """Average evaluation seeds within each model before summarizing training-seed deltas."""
    grouped = {}
    for row in rows:
        key = (row['geometry'], row['fraction'], row['training_seed'], row['variant'])
        grouped.setdefault(key, []).append(row['recovered_loss'])
    conditions = sorted({(r['geometry'], r['fraction']) for r in rows})
    summary = []
    for geometry, fraction in conditions:
        seeds = sorted({r['training_seed'] for r in rows if r['geometry'] == geometry and r['fraction'] == fraction})
        pairs = []
        for seed in seeds:
            growth = mean(grouped[(geometry, fraction, seed, 'growth')])
            damage = mean(grouped[(geometry, fraction, seed, 'damage')])
            pairs.append({'training_seed': seed, 'growth_loss': growth, 'damage_loss': damage,
                          'delta': damage - growth})
        deltas = [p['delta'] for p in pairs]
        summary.append({'geometry': geometry, 'fraction': fraction, 'training_seed_count': len(pairs),
                        'growth_mean': mean(p['growth_loss'] for p in pairs),
                        'damage_mean': mean(p['damage_loss'] for p in pairs),
                        'paired_delta_mean': mean(deltas),
                        'paired_delta_sd': stdev(deltas) if len(deltas) > 1 else None,
                        'pairs': pairs})
    return summary


def run_study(target, output, plan):
    # Validate the entire fixed plan before creating outputs or starting training.
    expected = {'training', 'training_seeds', 'evaluation_seeds', 'fractions', 'geometries',
                'grow_steps', 'recovery_steps'}
    if set(plan) != expected:
        raise ValueError('study plan must contain exactly the documented fields')
    config = TrainConfig(**plan['training'])
    if not config.pool_size or not config.damage_probability:
        raise ValueError('study training config must enable pool damage')
    for key in ('training_seeds', 'evaluation_seeds'):
        values = plan[key]
        if not values or any(type(s) is not int or s < 0 for s in values) or len(set(values)) != len(values):
            raise ValueError(f'{key} must be distinct nonnegative integers')
    if set(plan['training_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('training and evaluation seeds must be disjoint')
    if not plan['geometries'] or len(set(plan['geometries'])) != len(plan['geometries']) or any(g not in GEOMETRIES for g in plan['geometries']):
        raise ValueError('geometries must be distinct supported names')
    fractions = plan['fractions']
    if not fractions or any(not 0 <= f <= 1 for f in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError('fractions must be distinct values in [0, 1]')
    if any(type(plan[k]) is not int or plan[k] < 1 for k in ('grow_steps', 'recovery_steps')):
        raise ValueError('rollout steps must be positive integers')
    target_bytes = Path(target).read_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', {'status': 'running'})
    try:
        (output / 'target-source').write_bytes(target_bytes)
        write_json(output / 'plan.json', plan)
        write_json(output / 'manifest.json', {
            'target_sha256': hashlib.sha256(target_bytes).hexdigest(),
            'plan_sha256': hashlib.sha256((output / 'plan.json').read_bytes()).hexdigest(),
            'contrast': 'damage probability enabled versus zero; all other per-seed settings matched',
            'replication_unit': 'training seed; evaluation seeds averaged within model',
            'scope': 'fixed exploratory same-target study, not held-out generalization'})
        rows = []
        for seed in plan['training_seeds']:
            for variant in ('growth', 'damage'):
                settings = replace(config, seed=seed, damage_probability=0 if variant == 'growth' else config.damage_probability)
                run = output / f'{variant}-{seed}'
                print(f'study: {variant}, training seed {seed}', flush=True)
                train(output / 'target-source', run, settings)
                for geometry in plan['geometries']:
                    metrics = evaluate_recovery(run / 'checkpoint.pt', output / 'target-source',
                                                output / f'eval-{variant}-{seed}-{geometry}',
                                                grow_steps=plan['grow_steps'], recovery_steps=plan['recovery_steps'],
                                                fractions=fractions, seeds=plan['evaluation_seeds'], geometry=geometry)
                    rows.extend(dict(row, training_seed=seed, variant=variant) for row in metrics)
                    write_json(output / 'rows.json', rows)
        summary = summarize(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete'})
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
