import json

from PIL import Image
import pytest
import torch

from ncap.config import TrainConfig
from ncap.losses import image_loss
from ncap.simulate import export_rollout, load_checkpoint
from ncap.state import load_target
from ncap.trainer import train
from ncap.visualize import render_state


def test_target_alpha_padding_and_aspect(tmp_path):
    path = tmp_path / 'target.png'
    Image.new('RGBA', (8, 4), (200, 100, 50, 128)).save(path)
    target = load_target(path, 12, 2)
    assert target.shape == (1, 4, 12, 12)
    assert target[:, :, :4].count_nonzero() == 0
    assert target[:, :, 8:].count_nonzero() == 0
    assert target[:, :, :, :2].count_nonzero() == 0
    assert target[0, 0, 5, 5].item() == pytest.approx(200 / 255 * 128 / 255, abs=0.005)
    Image.new('RGBA', (8, 4), (255, 0, 0, 0)).save(path)
    assert load_target(path, 12, 2).count_nonzero() == 0


def test_loss_gradient_and_hidden_channels():
    state = torch.zeros(2, 8, 3, 3, requires_grad=True)
    target = torch.ones(1, 4, 3, 3)
    loss = image_loss(state, target)
    assert loss.item() == 1
    loss.backward()
    assert state.grad[:, :4].abs().sum() > 0
    assert state.grad[:, 4:].count_nonzero() == 0
    with pytest.raises(ValueError):
        image_loss(state, torch.ones(1, 4, 2, 3))


def test_render_composites_on_white():
    state = torch.zeros(1, 4, 2, 2)
    state[:, 0] = 0.5
    state[:, 3] = 0.5
    assert render_state(state).getpixel((0, 0)) == (255, 128, 128)


def test_training_checkpoint_and_immutable_replay(tmp_path):
    path = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 180, 0, 255)).save(path)
    config = TrainConfig(channels=4, hidden_size=8, size=5, padding=2, batch_size=1,
                         iterations=3, min_steps=2, max_steps=3, eval_steps=3)
    output = tmp_path / 'run'
    metrics = train(path, output, config)
    assert metrics['final_loss'] < metrics['seed_loss']
    assert json.loads((output / 'status.json').read_text())['status'] == 'complete'
    assert (output / 'target-source').read_bytes() == path.read_bytes()
    assert len((output / 'loss.csv').read_text().splitlines()) == 4
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    assert checkpoint['iteration'] == 3
    model, loaded = load_checkpoint(output / 'checkpoint.pt')
    replay = tmp_path / 'replay'
    replay.mkdir()
    export_rollout(model, size=loaded['size'], steps=loaded['eval_steps'],
                   seed=loaded['eval_seed'], output=replay)
    assert (replay / 'final.png').read_bytes() == (output / 'final.png').read_bytes()
    before = (output / 'checkpoint.pt').read_bytes()
    with pytest.raises(FileExistsError):
        train(path, output, config)
    assert before == (output / 'checkpoint.pt').read_bytes()
    repeat = tmp_path / 'repeat'
    repeated_metrics = train(path, repeat, config)
    assert repeated_metrics['final_loss'] == metrics['final_loss']
    assert (repeat / 'loss.csv').read_bytes() == (output / 'loss.csv').read_bytes()


def test_failed_run_is_preserved(tmp_path):
    path = tmp_path / 'empty.png'
    Image.new('RGBA', (2, 2)).save(path)
    output = tmp_path / 'failed'
    with pytest.raises(ValueError, match='visible'):
        train(path, output, TrainConfig())
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
    assert (output / 'target-source').exists()


@pytest.mark.parametrize('values', [{'padding': 32}, {'iterations': 0}, {'min_steps': 100},
                                    {'learning_rate': float('nan')}, {'size': 8.5}])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        TrainConfig(**values)
