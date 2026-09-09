"""Matched training-horizon pilot with fixed evaluation and explicit compute accounting."""

import csv
from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean, stdev

from .config import TrainConfig
from .evaluate import evaluate
from .trainer import train, write_json


def summarize_horizons(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['step'], row['training_seed'], row['arm']), []).append(row)
    result = []
    for step in sorted({r['step'] for r in rows}):
        for metric in ('loss', 'alpha_iou', 'foreground_cells'):
            pairs = []
            for seed in sorted({r['training_seed'] for r in rows}):
                short = mean(r[metric] for r in grouped[(step, seed, 'short')])
                long = mean(r[metric] for r in grouped[(step, seed, 'long')])
                pairs.append(dict(training_seed=seed, short=short, long=long, delta=long-short))
            result.append(dict(step=step, metric=metric, training_seed_count=len(pairs),
                               short_mean=mean(p['short'] for p in pairs),
                               long_mean=mean(p['long'] for p in pairs),
                               paired_delta_mean=mean(p['delta'] for p in pairs),
                               paired_delta_sd=stdev(p['delta'] for p in pairs) if len(pairs) > 1 else None,
                               pairs=pairs))
    return result


def validate_plan(plan):
    if set(plan) != {'training', 'training_seeds', 'evaluation_seeds', 'horizons', 'arms'}:
        raise ValueError('unexpected or missing horizon study fields')
    config = TrainConfig(**plan['training'])
    for key in ('training_seeds', 'evaluation_seeds', 'horizons'):
        values = plan[key]
        if not values or any(type(v) is not int or v < (1 if key == 'horizons' else 0) for v in values) or len(set(values)) != len(values):
            raise ValueError(f'invalid {key}')
    if set(plan['training_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('training and evaluation seeds must be disjoint')
    if set(plan['arms']) != {'short', 'long'}:
        raise ValueError('exactly short and long arms are required')
    settings = {}
    for arm, values in plan['arms'].items():
        if set(values) != {'min_steps', 'max_steps'}:
            raise ValueError('arms may change only training rollout lengths')
        settings[arm] = replace(config, **values)
    if settings['long'].min_steps <= settings['short'].max_steps:
        raise ValueError('long rollouts must start beyond the short rollout range')
    if max(plan['horizons']) <= settings['long'].max_steps:
        raise ValueError('evaluation must extend beyond both training ranges')
    return settings


def run_horizon_study(target, output, plan):
    settings = validate_plan(plan)
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
            'contrast': 'training rollout range only; matched optimizer iterations, not compute',
            'replication_unit': 'training seed; evaluation seeds averaged within model',
            'scope': 'fixed same-target pilot, no checkpoint selection or stability threshold'})
        rows, costs = [], []
        for seed in plan['training_seeds']:
            for arm in ('short', 'long'):
                config = replace(settings[arm], seed=seed)
                run = output / f'{arm}-{seed}'
                print(f'horizon study: {arm}, training seed {seed}', flush=True)
                metrics = train(output / 'target-source', run, config)
                with (run / 'loss.csv').open() as stream:
                    updates = sum(int(r['rollout_steps']) for r in csv.DictReader(stream))
                costs.append(dict(arm=arm, training_seed=seed, iterations=config.iterations,
                                  batch_size=config.batch_size, rollout_updates=updates,
                                  sample_updates=updates * config.batch_size,
                                  training_and_export_seconds=metrics['seconds']))
                write_json(output / 'costs.json', costs)
                observations = evaluate(run / 'checkpoint.pt', output / 'target-source',
                                        output / f'eval-{arm}-{seed}', horizons=plan['horizons'],
                                        seeds=plan['evaluation_seeds'])
                rows.extend(dict(row, arm=arm, training_seed=seed) for row in observations)
                write_json(output / 'rows.json', rows)
        summary = summarize_horizons(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete'})
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
