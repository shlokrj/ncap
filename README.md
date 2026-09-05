# ncap

Neural Cellular Automata Playground — exploring how a shared local neural rule can learn growth, persistence, and regeneration.

## Status

The initial PyTorch core is implemented: grid state, Sobel perception, stochastic neural updates, alive-cell masking, and differentiable multi-step rollout. Training, regeneration, experiments, and the interactive playground are planned. No trained growth or regeneration results are available yet.

## Getting started

Python 3.11 or newer is required, with a compatible PyTorch build.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

```python
import torch
from ncap import NeuralCellularAutomata, create_seed, state_to_rgba

model = NeuralCellularAutomata(channels=16)
seed = create_seed(height=64, width=64)
state = model.rollout(seed, steps=64, generator=torch.Generator().manual_seed(0))
rgba = state_to_rgba(state)  # [1, 4, 64, 64], unclamped
```

The update network starts with a zero output layer, so an untrained model preserves the seed. Learning a target is the next milestone.

## Core

States use `[batch, channels, height, width]`: RGB, alpha, then hidden channels. By default, 16 channels hold four visible and twelve hidden values. The seed activates alpha and hidden state in the center cell.

Each cell perceives its own state and normalized Sobel X/Y gradients. A shared two-layer network predicts a residual update, applied independently to cells with probability 0.5. A cell survives only if its 3×3 neighborhood contains alpha greater than 0.1 both before and after the update. Perception uses zero padding at grid boundaries.

| File | Purpose |
| --- | --- |
| `src/ncap/state.py` | Seeds and visible-channel conversion |
| `src/ncap/perception.py` | Fixed identity and Sobel filters |
| `src/ncap/model.py` | Shared update rule and rollout |
| `tests/test_core.py` | Numerical invariants and gradient checks |

## Next milestones

1. Load one RGBA target and train growth from a seed.
2. Add state pool training and evaluate persistence.
3. Train and measure recovery after damage.
4. Run controlled robustness experiments.
5. Build an interactive playground around validated models.

Generated runs, checkpoints, and exports belong under `data/` and are excluded from Git.
