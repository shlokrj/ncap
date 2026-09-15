import json
from dataclasses import asdict

import pytest
from PIL import Image

from ncap.budget_study import run_budget_study
from ncap.config import TrainConfig
from ncap.report import export_budget_report


@pytest.fixture
def study(tmp_path):
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 150, 0, 255)).save(target)
    config = TrainConfig(channels=4, hidden_size=8, size=5, padding=2,
                         batch_size=1, iterations=2, min_steps=1, max_steps=1,
                         eval_steps=1)
    plan = dict(training=asdict(config), training_seeds=[0], evaluation_seeds=[10],
                horizons=[1, 3], budgets=[1, 2],
                recovery=dict(grow_steps=1, recovery_steps=2,
                              fractions=[0, .25, 1], geometries=['center']))
    output = tmp_path / 'study'
    run_budget_study(target, output, plan)
    return output


def test_report_preserves_all_evidence_and_is_portable(study, tmp_path):
    first, second = tmp_path / 'report', tmp_path / 'again'
    export_budget_report(study, first)
    export_budget_report(study, second)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    manifest = json.loads((first / 'manifest.json').read_text())
    assert manifest['persistence_rows'] == 4
    assert manifest['recovery_rows'] == 6
    assert len(list(first.glob('seed-*.html'))) == 2
    html = (first / 'seed-0-budget-1.html').read_text()
    assert html.count('data:image/png;base64,') == 15
    assert 'Raw max |state|' in html and 'Foreground removed:' in html
    assert str(study) not in html
    assert json.loads((first / 'rows.json').read_text()) == json.loads((study / 'rows.json').read_text())
    with pytest.raises(FileExistsError):
        export_budget_report(study, first)


@pytest.mark.parametrize('fault', ['missing_image', 'incomplete', 'duplicate', 'changed_metric', 'wrong_checkpoint', 'wrong_plan'])
def test_report_rejects_inconsistent_evidence_before_writing(study, tmp_path, fault):
    def change(name, edit):
        path = study / name
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))
    if fault == 'missing_image':
        (study / 'eval-0-1/seed-10-step-1.png').unlink()
    elif fault == 'incomplete':
        change('status.json', lambda d: d.update(status='failed'))
    elif fault == 'duplicate':
        change('eval-0-1/metrics.json', lambda d: d.append(d[0]))
    elif fault == 'changed_metric':
        change('rows.json', lambda d: d[0].update(loss=99))
    elif fault == 'wrong_checkpoint':
        change('recovery-0-1-center/evaluation.json', lambda d: d.update(checkpoint_sha256='wrong'))
    else:
        change('plan.json', lambda d: d.update(horizons=[1, 4]))
    output = tmp_path / 'report'
    with pytest.raises((ValueError, FileNotFoundError)):
        export_budget_report(study, output)
    assert not output.exists()
