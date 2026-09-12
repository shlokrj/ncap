"""Fixed state-limit comparison, with shape quality measured separately from boundedness."""

from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean, stdev

from .config import TrainConfig
from .evaluate import evaluate
from .trainer import train, write_json


def summarize_bounds(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['step'], row['training_seed'], row['arm']), []).append(row)
    result = []
    for step in sorted({r['step'] for r in rows}):
        for metric in ('loss', 'alpha_iou', 'foreground_cells', 'state_abs_max'):
            pairs = []
            for seed in sorted({r['training_seed'] for r in rows}):
                unbounded = mean(r[metric] for r in grouped[(step, seed, 'unbounded')])
                bounded = mean(r[metric] for r in grouped[(step, seed, 'bounded')])
                pairs.append(dict(training_seed=seed, unbounded=unbounded, bounded=bounded,
                                  delta=bounded-unbounded))
            result.append(dict(step=step, metric=metric, training_seed_count=len(pairs),
                               unbounded_mean=mean(p['unbounded'] for p in pairs),
                               bounded_mean=mean(p['bounded'] for p in pairs),
                               paired_delta_mean=mean(p['delta'] for p in pairs),
                               paired_delta_sd=stdev(p['delta'] for p in pairs) if len(pairs)>1 else None,
                               pairs=pairs))
    return result


def run_bound_study(target, output, plan):
    if set(plan) != {'training', 'training_seeds', 'evaluation_seeds', 'horizons', 'state_limit'}:
        raise ValueError('unexpected or missing state-limit study fields')
    config = TrainConfig(**plan['training'])
    if config.state_limit is not None or plan['state_limit'] is None:
        raise ValueError('base training must be unbounded and study limit must be set')
    bounded = replace(config, state_limit=plan['state_limit'])
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
            'contrast': 'only state_limit changes; same iterations and sampled training horizons',
            'replication_unit': 'training seed; evaluation seeds averaged within model',
            'scope': 'same-target pilot; bounded values alone do not establish stable morphology'})
        rows = []
        for seed in plan['training_seeds']:
            for arm, settings in [('unbounded', config), ('bounded', bounded)]:
                run = output / f'{arm}-{seed}'
                print(f'state-limit study: {arm}, training seed {seed}', flush=True)
                train(output / 'target-source', run, replace(settings, seed=seed))
                metrics = evaluate(run / 'checkpoint.pt', output / 'target-source',
                                   output / f'eval-{arm}-{seed}', horizons=plan['horizons'],
                                   seeds=plan['evaluation_seeds'])
                rows.extend(dict(row, arm=arm, training_seed=seed) for row in metrics)
                write_json(output / 'rows.json', rows)
        summary = summarize_bounds(rows)
        write_json(output / 'summary.json', summary)
        write_json(output / 'status.json', {'status': 'complete'})
        return summary
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
