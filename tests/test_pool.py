import json
import pytest
import torch
from PIL import Image
from ncap.pool import StatePool
from ncap.state import create_seed
from ncap.config import TrainConfig
from ncap.trainer import train
from ncap.evaluate import evaluate
from ncap.simulate import load_checkpoint
from ncap.losses import image_loss
from ncap.state import load_target


def test_pool_sampling_reset_and_detach():
    seed = create_seed(channels=4, height=5, width=5)
    pool = StatePool(seed, 4)
    pool.states[:, 0] = torch.arange(1, 5)[:, None, None]
    indices, batch = pool.sample(4, seed, generator=torch.Generator().manual_seed(5))
    worst = (indices == 3).nonzero().item()
    assert torch.equal(batch[worst], seed[0])
    assert len(indices.unique()) == 4
    assert pool.states[3, 0].eq(4).all()
    updated = batch.clone().requires_grad_() * 2
    pool.commit(indices, updated)
    assert pool.states.grad_fn is None
    assert torch.equal(pool.states[indices], updated.detach())
    with pytest.raises(ValueError):
        pool.commit(indices, torch.full_like(updated, float('nan')))


def test_pool_training_and_continuous_evaluation(tmp_path):
    target_path = tmp_path / 'target.png'
    Image.new('RGBA', (1, 1), (0, 160, 0, 255)).save(target_path)
    config = TrainConfig(channels=4, hidden_size=8, size=5, padding=2, batch_size=2,
                         iterations=4, min_steps=2, max_steps=3, eval_steps=3, pool_size=4)
    run = tmp_path / 'run'
    train(target_path, run, config)
    checkpoint = torch.load(run / 'checkpoint.pt', weights_only=True)
    assert checkpoint['pool'].shape == (4, 4, 5, 5)
    assert checkpoint['pool'].grad_fn is None
    repeat = tmp_path / 'repeat'
    train(target_path, repeat, config)
    repeated = torch.load(repeat / 'checkpoint.pt', weights_only=True)
    assert torch.equal(checkpoint['pool'], repeated['pool'])
    results = evaluate(run / 'checkpoint.pt', target_path, tmp_path / 'evaluation', [3, 8], [10, 11])
    assert len(results) == 4
    model, _ = load_checkpoint(run / 'checkpoint.pt')
    state = model.rollout(create_seed(channels=4, height=5, width=5), 8,
                          generator=torch.Generator().manual_seed(10))
    target = load_target(target_path, 5, 2)
    assert results[1]['loss'] == pytest.approx(image_loss(state, target).item())
    assert all(0 <= row['alpha_iou'] <= 1 for row in results)
    assert json.loads((tmp_path / 'evaluation/status.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        evaluate(run / 'checkpoint.pt', target_path, tmp_path / 'evaluation', [3], [10])


def test_invalid_pool_config():
    with pytest.raises(ValueError):
        TrainConfig(pool_size=2, batch_size=4)
