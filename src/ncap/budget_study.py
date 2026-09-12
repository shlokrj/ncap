"""Compare fixed training budgets along matched trajectories, without checkpoint selection."""

import csv
from dataclasses import replace
import hashlib
from pathlib import Path
from statistics import mean, stdev
import torch

from .config import TrainConfig
from .evaluate import evaluate
from .trainer import train, write_json


def summarize_budgets(rows):
    budgets = sorted({r['budget'] for r in rows})
    seeds = sorted({r['training_seed'] for r in rows})
    grouped = {}
    for row in rows:
        grouped.setdefault((row['step'], row['training_seed'], row['budget']), []).append(row)
    result = []
    for step in sorted({r['step'] for r in rows}):
        for metric in ('loss', 'alpha_iou', 'foreground_cells', 'state_abs_max'):
            for budget in budgets:
                pairs = []
                for seed in seeds:
                    base = mean(r[metric] for r in grouped[(step, seed, budgets[0])])
                    current = mean(r[metric] for r in grouped[(step, seed, budget)])
                    pairs.append(dict(training_seed=seed, baseline=base, value=current, delta=current-base))
                result.append(dict(step=step, metric=metric, budget=budget, baseline_budget=budgets[0],
                                   training_seed_count=len(pairs), mean=mean(p['value'] for p in pairs),
                                   paired_delta_mean=mean(p['delta'] for p in pairs),
                                   paired_delta_sd=stdev(p['delta'] for p in pairs) if len(pairs)>1 else None,
                                   pairs=pairs))
    return result


def run_budget_study(target, output, plan):
    if set(plan) != {'training','training_seeds','evaluation_seeds','horizons','budgets'}:
        raise ValueError('unexpected or missing budget study fields')
    config = TrainConfig(**plan['training'])
    for key in ('training_seeds','evaluation_seeds','horizons','budgets'):
        values = plan[key]
        if not values or any(type(v) is not int or v < (0 if key.endswith('seeds') else 1) for v in values) or len(set(values))!=len(values):
            raise ValueError(f'invalid {key}')
    budgets = sorted(plan['budgets'])
    if len(budgets)<2 or budgets[-1]!=config.iterations:
        raise ValueError('at least two budgets required, ending at training.iterations')
    if set(plan['training_seeds']) & set(plan['evaluation_seeds']):
        raise ValueError('training and evaluation seeds must be disjoint')
    if max(plan['horizons'])<=config.max_steps:
        raise ValueError('evaluation must extend beyond the training rollout range')
    target_bytes=Path(target).read_bytes()
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',{'status':'running'})
    try:
        (output/'target-source').write_bytes(target_bytes)
        write_json(output/'plan.json',plan)
        write_json(output/'manifest.json',{
            'target_sha256':hashlib.sha256(target_bytes).hexdigest(),
            'plan_sha256':hashlib.sha256((output/'plan.json').read_bytes()).hexdigest(),
            'contrast':'fixed iteration budgets on the same training trajectory; cumulative costs',
            'replication_unit':'training seed; checkpoints from one seed are paired, not independent',
            'scope':'same-target diagnostic; all predeclared checkpoints evaluated without selection'})
        rows,costs=[],[]
        for seed in plan['training_seeds']:
            run=output/f'train-{seed}'
            print(f'budget study: training seed {seed}',flush=True)
            train(output/'target-source',run,replace(config,seed=seed),checkpoint_iterations=budgets[:-1])
            with (run/'loss.csv').open() as stream:
                log=list(csv.DictReader(stream))
            for budget in budgets:
                checkpoint=run/('checkpoint.pt' if budget==budgets[-1] else f'checkpoint-{budget}.pt')
                saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
                updates=sum(int(r['rollout_steps']) for r in log[:budget])
                costs.append(dict(training_seed=seed,budget=budget,rollout_updates=updates,
                                  sample_updates=updates*config.batch_size,training_seconds=saved['training_seconds'],
                                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
                write_json(output/'costs.json',costs)
                metrics=evaluate(checkpoint,output/'target-source',output/f'eval-{seed}-{budget}',
                                 horizons=plan['horizons'],seeds=plan['evaluation_seeds'])
                rows.extend(dict(row,training_seed=seed,budget=budget) for row in metrics)
                write_json(output/'rows.json',rows)
        result=summarize_budgets(rows)
        write_json(output/'summary.json',result)
        write_json(output/'status.json',{'status':'complete'})
        return result
    except BaseException as exc:
        write_json(output/'status.json',{'status':'failed','error':str(exc)})
        raise
