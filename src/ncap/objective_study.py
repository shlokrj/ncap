"""Fixed excess-state objective comparison with unclamped inference."""

from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean, stdev

from .config import TrainConfig
from .evaluate import evaluate
from .trainer import train, write_json


def summarize_objective(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['step'], row['training_seed'], row['arm']), []).append(row)
    result = []
    for step in sorted({r['step'] for r in rows}):
        for metric in ('loss', 'alpha_iou', 'foreground_cells', 'state_abs_max'):
            pairs = []
            for seed in sorted({r['training_seed'] for r in rows}):
                baseline = mean(r[metric] for r in grouped[(step, seed, 'baseline')])
                regularized = mean(r[metric] for r in grouped[(step, seed, 'regularized')])
                pairs.append(dict(training_seed=seed, baseline=baseline, regularized=regularized,
                                  delta=regularized-baseline))
            result.append(dict(step=step, metric=metric, training_seed_count=len(pairs),
                               baseline_mean=mean(p['baseline'] for p in pairs),
                               regularized_mean=mean(p['regularized'] for p in pairs),
                               paired_delta_mean=mean(p['delta'] for p in pairs),
                               paired_delta_sd=stdev(p['delta'] for p in pairs) if len(pairs)>1 else None,
                               pairs=pairs))
    return result


def run_objective_study(target, output, plan):
    if set(plan) != {'training', 'training_seeds', 'evaluation_seeds', 'horizons', 'excess_weight'}:
        raise ValueError('unexpected or missing excess-state study fields')
    config = TrainConfig(**plan['training'])
    if config.excess_weight != 0 or config.state_limit is not None or plan['excess_weight'] <= 0:
        raise ValueError('base training must disable penalty and clamping; study weight must be positive')
    regularized = replace(config, excess_weight=plan['excess_weight'])
    for key in ('training_seeds', 'evaluation_seeds', 'horizons'):
        values = plan[key]
        if not values or any(type(v) is not int or v < (1 if key == 'horizons' else 0) for v in values) or len(set(values)) != len(values):
            raise ValueError(f'invalid {key}')
    if set(plan['training_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('training and evaluation seeds must be disjoint')
    if max(plan['horizons']) <= config.max_steps:
        raise ValueError('evaluation must extend beyond the training horizon')
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
            'contrast': 'only excess_weight changes; same iterations and sampled training horizons',
            'replication_unit': 'training seed; evaluation seeds averaged within model',
            'scope': 'same-target pilot; reduced state magnitude alone does not establish stable morphology'})
        rows = []
        for seed in plan['training_seeds']:
            for arm, settings in [('baseline', config), ('regularized', regularized)]:
                run = output / f'{arm}-{seed}'
                print(f'excess-state study: {arm}, training seed {seed}', flush=True)
                train(output / 'target-source', run, replace(settings, seed=seed))
                metrics = evaluate(run / 'checkpoint.pt', output / 'target-source',
                                   output / f'eval-{arm}-{seed}', horizons=plan['horizons'],
                                   seeds=plan['evaluation_seeds'])
                rows.extend(dict(row, arm=arm, training_seed=seed) for row in metrics)
                write_json(output / 'rows.json', rows)
        summary = summarize_objective(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete'})
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
