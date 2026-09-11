from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from ncap.config import TrainConfig
from ncap.experiments import run_study
from ncap.persistence import run_persistence, summarize_persistence


@pytest.fixture
def study(tmp_path):
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 150, 0, 255)).save(target)
    plan = dict(training=asdict(TrainConfig(channels=4, hidden_size=8, size=5, padding=2,
                batch_size=2, iterations=3, min_steps=2, max_steps=3, eval_steps=3,
                pool_size=2, damage_probability=.5)), training_seeds=[0,1], evaluation_seeds=[10],
                fractions=[0], geometries=['dropout'], grow_steps=3, recovery_steps=3)
    source = tmp_path / 'study'
    run_study(target, source, plan)
    return source


def test_checkpoint_reuse_and_summary(study, tmp_path):
    plan = {'horizons': [3, 8], 'evaluation_seeds': [20, 21]}
    output = tmp_path / 'persistence'
    summary = run_persistence(study, output, plan)
    assert len(summary) == 6
    assert len(json.loads((output / 'rows.json').read_text())) == 16
    assert all(r['training_seed_count'] == 2 for r in summary)
    assert all(r['growth_change_mean'] == r['damage_change_mean'] == 0 for r in summary if r['step'] == 3)
    assert (output / 'checkpoints/growth-0.pt').read_bytes() == (study / 'growth-0/checkpoint.pt').read_bytes()
    assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        run_persistence(study, output, plan)


def test_mismatched_checkpoint_rejected(study, tmp_path):
    path = study / 'damage-0/checkpoint.pt'
    checkpoint = torch.load(path, weights_only=True)
    checkpoint['config']['seed'] = 99
    torch.save(checkpoint, path)
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='settings'):
        run_persistence(study, output, {'horizons': [3], 'evaluation_seeds': [20]})
    assert not output.exists()


def test_tampered_target_rejected(study, tmp_path):
    (study / 'target-source').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        run_persistence(study, tmp_path / 'rejected', {'horizons': [3], 'evaluation_seeds': [20]})


def test_aggregation_pairs_training_seeds():
    rows = []
    for training_seed, count, value in [(0, 1, 2), (1, 3, 6)]:
        for _ in range(count):
            for step in [1, 2]:
                for variant in ['growth','damage']:
                    metric = value * step if variant == 'growth' else value
                    rows.append(dict(training_seed=training_seed, variant=variant, step=step,
                                     loss=metric, alpha_iou=metric, foreground_cells=metric))
    row = next(r for r in summarize_persistence(rows) if r['step'] == 2 and r['metric'] == 'loss')
    assert row['growth_mean'] == 8
    assert row['damage_mean'] == 4
    assert row['growth_change_mean'] == 4
    assert row['paired_delta_mean'] == -4


def test_evaluation_preserves_earlier_measurements(tmp_path, monkeypatch):
    import ncap.evaluate as module
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 150, 0, 255)).save(target)
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    class FailingModel:
        calls = 0
        def rollout(self, state, steps, generator):
            self.calls += 1
            return state if self.calls == 1 else state * float('nan')
    monkeypatch.setattr(module, 'load_checkpoint', lambda path: (FailingModel(), dict(threads=1, size=5, padding=2, channels=4)))
    output = tmp_path / 'evaluation'
    with pytest.raises(FloatingPointError):
        module.evaluate(checkpoint, target, output, horizons=[1,2], seeds=[20])
    assert len(json.loads((output / 'metrics.json').read_text())) == 1
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'


def test_source_checkpoint_without_new_optional_field(study, tmp_path):
    for name in ['growth-0', 'damage-0', 'growth-1', 'damage-1']:
        path = study / name / 'checkpoint.pt'
        checkpoint = torch.load(path, weights_only=True)
        for key in ('state_limit', 'excess_weight', 'excess_threshold'):
            checkpoint['config'].pop(key)
        torch.save(checkpoint, path)
    summary = run_persistence(study, tmp_path / 'legacy',
                              {'horizons': [3, 8], 'evaluation_seeds': [20]})
    assert len(summary) == 6
