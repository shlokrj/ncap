import pytest
import torch

from ncap import NeuralCellularAutomata, SobelPerception, create_seed, rgba_to_state, state_to_rgba


def test_seed_and_visible_conversion():
    seed = create_seed(2, 8, 7, 9, dtype=torch.float64)
    assert seed.shape == (2, 8, 7, 9)
    assert seed.dtype == torch.float64
    assert seed[:, :3].count_nonzero() == 0
    assert torch.all(seed[:, 3:, 3, 4] == 1)
    assert seed.count_nonzero() == 10
    rgba = state_to_rgba(seed).clone().requires_grad_()
    restored = rgba_to_state(rgba, 8)
    assert torch.equal(restored[:, :4], rgba)
    assert restored[:, 4:].count_nonzero() == 0
    restored.sum().backward()
    assert torch.all(rgba.grad == 1)


def test_sobel_orientation_and_channel_isolation():
    state = torch.zeros(1, 4, 7, 9)
    state[:, 0] = torch.arange(9).float()
    state[:, 1] = torch.arange(7).float()[:, None]
    perceived = SobelPerception(4)(state)
    assert perceived.shape == (1, 12, 7, 9)
    assert torch.equal(perceived[:, 0], state[:, 0])
    assert torch.allclose(perceived[:, 1, 1:-1, 1:-1], torch.ones(1, 5, 7))
    assert torch.count_nonzero(perceived[:, 2, 1:-1, 1:-1]) == 0
    assert torch.allclose(perceived[:, 5, 1:-1, 1:-1], torch.ones(1, 5, 7))
    assert perceived[:, 6:].count_nonzero() == 0


def test_initial_rule_and_gradients():
    model = NeuralCellularAutomata(8, 16, fire_rate=1)
    seed = create_seed(2, 8, 9, 11)
    result = model.rollout(seed, 3)
    assert result.shape == seed.shape
    assert torch.isfinite(result).all()
    assert torch.equal(result, seed)
    result[:, :3].sum().backward()
    assert model.update[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_alive_neighborhood_and_dead_cells():
    model = NeuralCellularAutomata(4, 8, fire_rate=1)
    seed = create_seed(1, 4, 7, 7)
    assert model.alive_mask(seed).sum() == 9
    with torch.no_grad():
        model.update[-1].bias.fill_(1)
    assert model(torch.zeros_like(seed)).count_nonzero() == 0
    result = model(seed)
    assert result[0, 3].count_nonzero() == 9
    with torch.no_grad():
        model.update[-1].bias[3] = -2
    assert model(seed).count_nonzero() == 0


def test_fire_rate_and_reproducibility():
    model = NeuralCellularAutomata(4, 8, fire_rate=0)
    state = torch.ones(2, 4, 8, 8)
    with torch.no_grad():
        model.update[-1].bias.fill_(0.2)
    assert torch.equal(model(state), state)
    model.fire_rate = 0.5
    a = model(state, generator=torch.Generator().manual_seed(7))
    b = model(state, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert (a == state).any() and (a > state).any()
    assert torch.equal(a[:, 0] - state[:, 0], a[:, 3] - state[:, 3])


@pytest.mark.parametrize('kwargs', [{'channels': 3}, {'height': 0}, {'batch_size': 0}])
def test_invalid_seed(kwargs):
    with pytest.raises(ValueError):
        create_seed(**kwargs)


def test_invalid_model_inputs():
    with pytest.raises(ValueError):
        NeuralCellularAutomata(fire_rate=1.1)
    model = NeuralCellularAutomata()
    with pytest.raises(ValueError):
        model(create_seed(channels=4))
    with pytest.raises(ValueError):
        model.rollout(create_seed(), -1)
    seed = create_seed()
    assert model.rollout(seed, 0) is seed
