from dataclasses import asdict
import json
import pytest
import torch
from PIL import Image
from ncap.config import TrainConfig
from ncap.damage import apply_damage, GEOMETRIES
from ncap.experiments import run_study, summarize


@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_exact_geometry_masks(geometry):
    state = torch.ones(2, 8, 8, 10)
    damaged = apply_damage(state, .25, geometry, generator=torch.Generator().manual_seed(3))
    assert torch.all((damaged[:, 0] == 0).sum((1, 2)) == 20)
    assert torch.equal(damaged[:, 0], damaged[:, 7])
    assert torch.equal(apply_damage(state, 0, geometry, generator=torch.Generator()), state)
    assert apply_damage(state, 1, geometry, generator=torch.Generator()).count_nonzero() == 0
    if geometry == 'edge':
        assert damaged[:, :, :, :2].count_nonzero() == 0
    if geometry == 'horizontal':
        assert damaged[:, :, 3:5].count_nonzero() == 0
    if geometry == 'center':
        assert damaged[:, :, 3:5, 4:6].count_nonzero() == 0


def test_summary_uses_training_seed_pairs():
    rows = []
    for training_seed, count, delta in [(0, 1, -1), (1, 3, 3)]:
        for _ in range(count):
            for variant, loss in [('growth', 5), ('damage', 5 + delta)]:
                rows.append(dict(geometry='center', fraction=.25, training_seed=training_seed,
                                 variant=variant, recovered_loss=loss))
    result = summarize(rows)[0]
    assert result['paired_delta_mean'] == 1
    assert result['training_seed_count'] == 2
    assert result['paired_delta_sd'] == pytest.approx(2 ** .5 * 2)


def test_study_end_to_end(tmp_path):
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 150, 0, 255)).save(target)
    plan = dict(training=asdict(TrainConfig(channels=4, hidden_size=8, size=5, padding=2,
                batch_size=2, iterations=3, min_steps=2, max_steps=3, eval_steps=3,
                pool_size=2, damage_probability=.5)), training_seeds=[0, 1], evaluation_seeds=[10],
                fractions=[0, .5], geometries=['dropout', 'center'], grow_steps=3,recovery_steps=3)
    output = tmp_path / 'study'
    result = run_study(target, output, plan)
    assert len(result) == 4
    assert all(r['training_seed_count'] == 2 for r in result)
    assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    growth = json.loads((output / 'growth-0/config.json').read_text())
    damage = json.loads((output / 'damage-0/config.json').read_text())
    assert {k for k in growth if growth[k] != damage[k]} == {'damage_probability'}
    assert (output / 'target-source').read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        run_study(target, output, plan)
    plan['evaluation_seeds'] = [0]
    with pytest.raises(ValueError):
        run_study(target, tmp_path / 'invalid', plan)
    assert not (tmp_path / 'invalid').exists()


def test_failed_study_retains_plan(tmp_path, monkeypatch):
    from ncap import experiments
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 150, 0, 255)).save(target)
    plan = dict(training=asdict(TrainConfig(pool_size=8, damage_probability=.5)),
                training_seeds=[0], evaluation_seeds=[10], fractions=[.25],
                geometries=['center'], grow_steps=3, recovery_steps=3)
    def fail(*args, **kwargs):
        raise RuntimeError('training failed')
    monkeypatch.setattr(experiments, 'train', fail)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='training failed'):
        run_study(target, output, plan)
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
    assert json.loads((output / 'plan.json').read_text()) == plan
    assert (output / 'target-source').read_bytes() == target.read_bytes()
