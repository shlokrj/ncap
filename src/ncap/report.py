"""Export complete budget-study evidence as portable, static HTML pages."""

import base64
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path

from PIL import Image


STYLE = '''<style>
body{font:15px system-ui,sans-serif;color:#222;max-width:1200px;margin:32px auto;padding:0 20px}
a{color:#245f96}table{border-collapse:collapse;margin:16px 0 32px}th,td{padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top}
img{width:96px;height:96px;object-fit:contain;image-rendering:pixelated;background:white}
small{display:block;line-height:1.6}.scroll{overflow-x:auto}summary{cursor:pointer;padding:12px 0}h2{margin-top:32px}
</style>'''


def _page(title, body):
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title>{STYLE}<body><h1>{escape(title)}</h1>{body}</body></html>')


def export_budget_report(study, output):
    """Validate all planned observations before creating a new report directory.

    Images are copied inline, never rerun or used to recompute raw-state metrics.
    Every consumed source file is hashed in the manifest.
    """
    study, output = Path(study), Path(output)
    if output.exists():
        raise FileExistsError(output)
    hashes = {}

    def read(name):
        data = (study / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        return data

    def document(name):
        return json.loads(read(name))

    def complete(directory=''):
        if document(f'{directory}status.json')['status'] != 'complete':
            raise ValueError('report requires a complete study and evaluations')

    def picture(name, label):
        data = read(name)
        with Image.open(BytesIO(data)) as image:
            if image.format != 'PNG':
                raise ValueError('evaluation images must be PNG')
            image.verify()
        return f'<img alt="{escape(label, quote=True)}" src="data:image/png;base64,{base64.b64encode(data).decode()}">'

    def number(value):
        return 'n/a' if value is None else format(value, '.5g')

    complete()
    plan = document('plan.json')
    manifest = document('manifest.json')
    if hashes['plan.json'] != manifest['plan_sha256']:
        raise ValueError('study plan hash mismatch')
    target = read('target-source')
    if hashlib.sha256(target).hexdigest() != manifest['target_sha256']:
        raise ValueError('study target hash mismatch')
    rows = document('rows.json')
    recovery_rows = document('recovery-rows.json') if 'recovery' in plan else []
    costs = document('costs.json')
    observed, recovered = [], []
    pages = {}
    links = []
    for training_seed in plan['training_seeds']:
        for budget in sorted(plan['budgets']):
            prefix = f'eval-{training_seed}-{budget}/'
            complete(prefix)
            metadata = document(prefix + 'evaluation.json')
            matches = [c for c in costs if c['training_seed'] == training_seed and c['budget'] == budget]
            if len(matches) != 1:
                raise ValueError('missing or duplicate checkpoint cost')
            checkpoint_hash = matches[0]['checkpoint_sha256']

            def check_metadata(meta):
                if (meta['checkpoint_sha256'] != checkpoint_hash
                        or meta['target_sha256'] != manifest['target_sha256']
                        or meta['config']['iterations'] != budget
                        or meta['config']['seed'] != training_seed
                        or meta['seeds'] != plan['evaluation_seeds']):
                    raise ValueError('evaluation provenance mismatch')

            check_metadata(metadata)
            if metadata['horizons'] != sorted(plan['horizons']):
                raise ValueError('evaluation horizons mismatch')
            metrics = document(prefix + 'metrics.json')
            expected = {(seed, step) for seed in plan['evaluation_seeds'] for step in plan['horizons']}
            if len(metrics) != len(expected) or {(r['seed'], r['step']) for r in metrics} != expected:
                raise ValueError('missing or duplicate persistence observations')
            observed.extend(dict(r, training_seed=training_seed, budget=budget) for r in metrics)
            title = f'Training seed {training_seed}, {budget} iterations'
            body = '<p><a href="index.html">All checkpoints</a></p><p>Images are display-clamped on white. Metrics use raw states. Low image error alone does not establish stability.</p>'
            body += '<p>Training target</p>' + picture(f'train-{training_seed}/target.png', 'Training target at model resolution')
            body += '<h2>Persistence</h2><p>Each row follows one continuous rollout.</p>'
            body += '<div class="scroll"><table><tr><th>Update seed</th>'
            body += ''.join(f'<th>{step} steps</th>' for step in sorted(plan['horizons'])) + '</tr>'
            for seed in plan['evaluation_seeds']:
                body += f'<tr><th>{seed}</th>'
                for step in sorted(plan['horizons']):
                    row = next(r for r in metrics if r['seed'] == seed and r['step'] == step)
                    body += '<td>' + picture(prefix + f'seed-{seed}-step-{step}.png', f'Seed {seed}, step {step}')
                    body += ''.join(f'<small>{label}: {number(row[key])}</small>' for key, label in
                                    [('loss', 'MSE'), ('alpha_iou', 'Alpha IoU'), ('foreground_cells', 'Foreground cells'), ('state_abs_max', 'Raw max |state|')]) + '</td>'
                body += '</tr>'
            body += '</table></div>'
            if 'recovery' in plan:
                recovery = plan['recovery']
                body += f'<h2>Recovery</h2><p>{recovery["grow_steps"]} growth steps, then {recovery["recovery_steps"]} recovery steps. Damage fractions refer to grid area. Controls use matching update masks.</p>'
                for geometry in recovery['geometries']:
                    prefix = f'recovery-{training_seed}-{budget}-{geometry}/'
                    complete(prefix)
                    meta = document(prefix + 'evaluation.json')
                    check_metadata(meta)
                    if any(meta[k] != recovery[k] for k in ('grow_steps', 'recovery_steps', 'fractions')) or meta['geometry'] != geometry:
                        raise ValueError('recovery settings mismatch')
                    metrics = document(prefix + 'metrics.json')
                    expected = {(seed, fraction, geometry) for seed in plan['evaluation_seeds'] for fraction in recovery['fractions']}
                    if len(metrics) != len(expected) or {(r['seed'], r['fraction'], r['geometry']) for r in metrics} != expected:
                        raise ValueError('missing or duplicate recovery observations')
                    recovered.extend(dict(r, training_seed=training_seed, budget=budget) for r in metrics)
                    body += f'<details><summary>{escape(geometry)}</summary><div class="scroll"><table><tr><th>Update seed / grid removed</th><th>Grown</th><th>Damaged</th><th>Recovered</th><th>Control</th></tr>'
                    for seed in plan['evaluation_seeds']:
                        for index, fraction in enumerate(recovery['fractions']):
                            row = next(r for r in metrics if r['seed'] == seed and r['fraction'] == fraction)
                            body += f'<tr><th>{seed} / {fraction:g}<small>Foreground removed: {number(row["foreground_removed_fraction"])}</small></th>'
                            for suffix, key in [('grown', 'grown_loss'), (f'damage-{index}', 'damaged_loss'), (f'recovery-{index}', 'recovered_loss'), ('control', 'control_loss')]:
                                body += '<td>' + picture(prefix + f'seed-{seed}-{suffix}.png', f'{geometry}, seed {seed}, fraction {fraction:g}, {suffix}')
                                body += f'<small>MSE: {number(row[key])}</small></td>'
                            body += '</tr>'
                    body += '</table></div></details>'
            name = f'seed-{training_seed}-budget-{budget}.html'
            pages[name] = _page(title, body)
            links.append(f'<li><a href="{name}">{title}</a></li>')
    # Compare whole records, not just counts, so aggregate metrics cannot silently diverge.
    canonical = lambda records: sorted(json.dumps(r, sort_keys=True, allow_nan=False) for r in records)
    if canonical(rows) != canonical(observed) or canonical(recovery_rows) != canonical(recovered):
        raise ValueError('aggregate rows disagree with evaluation metrics')
    pages['index.html'] = _page('Budget study', '<p>All planned seeds and checkpoints. No selection or new simulation. These are same-target diagnostics, not held-out evaluation.</p><ul>' + ''.join(links) + '</ul><p>Images are display-clamped. Raw metrics accompany each image. Source hashes and exact observations are in manifest.json, rows.json, and recovery-rows.json.</p>')
    output.mkdir(parents=True, exist_ok=False)
    for name, html in pages.items():
        (output / name).write_text(html)
    for name, value in [('rows.json', rows), ('recovery-rows.json', recovery_rows), ('plan.json', plan),
                        ('manifest.json', {'source_sha256': hashes, 'persistence_rows': len(rows), 'recovery_rows': len(recovery_rows),
                                           'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                           'files_sha256': {name: hashlib.sha256(html.encode()).hexdigest() for name, html in pages.items()}})]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return output / 'index.html'
