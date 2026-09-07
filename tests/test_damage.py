from dataclasses import replace
import torch
import pytest
from PIL import Image
from ncap.damage import circle_damage, rectangle_damage, cell_dropout
from ncap.config import TrainConfig
from ncap.trainer import train
from ncap.recovery import evaluate_recovery


def test_geometry_and_gradients():
    state = torch.ones(2, 8, 7, 7, requires_grad=True)
    circle = circle_damage(state, (3, 3), 1)
    assert (circle[0, 0] == 0).sum() == 5
    rectangle = rectangle_damage(state, 1, 2, 3, 4)
    assert (rectangle[0, 0] == 0).sum() == 12
    assert torch.equal(rectangle[:, 0], rectangle[:, 7])
    rectangle.sum().backward()
    assert torch.equal(state.grad, rectangle.detach())
    assert torch.all(state == 1)


def test_dropout_exact_and_repeatable():
    state = torch.ones(3, 8, 10, 10)
    a = cell_dropout(state, .25, generator=torch.Generator().manual_seed(7))
    b = cell_dropout(state, .25, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert torch.all((a[:, 0] == 0).sum(dim=(1, 2)) == 25)
    assert torch.equal(a[:, 0], a[:, 7])
    assert not torch.equal(a[0], a[1])
    assert torch.equal(cell_dropout(state, 0, generator=torch.Generator()), state)
    assert cell_dropout(state, 1, generator=torch.Generator()).count_nonzero() == 0
    with pytest.raises(ValueError):
        cell_dropout(state, float('nan'), generator=torch.Generator())
    with pytest.raises(ValueError):
        rectangle_damage(state, 9, 9, 2, 2)


def test_regeneration_training_and_paired_control(tmp_path):
    target = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 160, 0, 255)).save(target)
    config = TrainConfig(channels=4, hidden_size=8, size=5, padding=2, batch_size=2,
                         iterations=12, min_steps=2, max_steps=3, eval_steps=3,
                         pool_size=2, damage_probability=1, damage_fraction=.5)
    run = tmp_path / 'run'
    train(target, run, config)
    train(target, tmp_path / 'repeat', config)
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    repeated = torch.load(tmp_path / 'repeat/checkpoint.pt', weights_only=True)
    assert torch.equal(checkpoint['pool'], repeated['pool'])
    assert not torch.equal(checkpoint['damage_generator_state'], torch.Generator().manual_seed(0).get_state())
    rows = evaluate_recovery(run / 'checkpoint.pt', target, tmp_path / 'recovery',
                             grow_steps=3, recovery_steps=4, fractions=[0, .5, 1], seeds=[10])
    assert rows[0]['recovered_loss'] == rows[0]['control_loss']
    assert rows[0]['foreground_removed_fraction'] == 0
    assert rows[-1]['foreground_removed_fraction'] == 1
    assert rows[-1]['recovered_loss'] == rows[-1]['damaged_loss']
    with pytest.raises(FileExistsError):
        evaluate_recovery(run / 'checkpoint.pt', target, tmp_path / 'recovery')
    with pytest.raises(ValueError):
        replace(config, pool_size=0)
