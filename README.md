# ncap

Neural Cellular Automata Playground — exploring how a shared local neural rule can learn growth, persistence, and regeneration.

## Status

The PyTorch core and seed-only training workflow are implemented: load an RGBA target, train a shared update rule, save a checkpoint, and export a seeded rollout. State pools, regeneration, controlled experiments, and the playground are next. Small synthetic tests validate the workflow; stable target growth and regeneration have not yet been demonstrated.

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

The update network starts with a zero output layer, so an untrained model preserves the seed. Use the training workflow below to learn a target.

## Core

States use `[batch, channels, height, width]`: RGB, alpha, then hidden channels. By default, 16 channels hold four visible and twelve hidden values. The seed activates alpha and hidden state in the center cell.

Each cell perceives its own state and normalized Sobel X/Y gradients. A shared two-layer network predicts a residual update, applied independently to cells with probability 0.5. A cell survives only if its 3×3 neighborhood contains alpha greater than 0.1 both before and after the update. Perception uses zero padding at grid boundaries.

| File | Purpose |
| --- | --- |
| `src/ncap/state.py` | Seeds and visible-channel conversion |
| `src/ncap/perception.py` | Fixed identity and Sobel filters |
| `src/ncap/model.py` | Shared update rule and rollout |
| `tests/test_core.py` | Numerical invariants and gradient checks |

## Train one target

Supply a local image; no target images or pretrained checkpoints are bundled. Images are fitted into the configured square grid with aspect ratio preserved and transparent padding. The objective is mean squared error over premultiplied RGB and alpha, so invisible RGB values do not affect the loss. Exports composite the visible channels on white.

```sh
ncap-train --target /path/to/target.png --config configs/base.json \
  --output data/runs/first-target
ncap-simulate --checkpoint data/runs/first-target/checkpoint.pt \
  --steps 96 --seed 10000 --output data/exports/first-target
```

After installation, `python scripts/train.py` and `python scripts/simulate.py` accept the same options. `--steps` overrides training iterations for training, and means automaton updates for simulation. `--seed` sets the corresponding random seed. The base configuration trains for 8,000 iterations with 64–96 updates per iteration; this is a starting configuration, not a validated recipe or guaranteed convergence.

Both commands require a new output directory and refuse to overwrite an existing run. Failed runs retain their artifacts and a failure status. Training runs contain:

| Artifact | Contents |
| --- | --- |
| `config.json`, `environment.json` | Settings, seeds, runtime versions, source hashes, Git state, target hash |
| `target-source`, `target.png` | Original target bytes and fitted preview |
| `loss.csv` | Training iteration, sampled rollout length, and loss |
| `checkpoint.pt` | Model, optimizer, settings, iteration, and training RNG states |
| `metrics.json` | Seed loss and final loss for one separately seeded rollout |
| `final.png`, `growth.gif` | Final state and sampled rollout frames on white |
| `status.json` | Running, complete, or failed status |

Training currently runs on CPU and starts every batch from fresh seeds. The final metric uses the training target and one evaluation seed; it is a diagnostic, not held-out evidence or a stability test. Checkpoints support simulation; a resume-training command is not implemented. Exact replay is tested within the same runtime, not promised across platforms or PyTorch versions.

## Next milestones

1. Validate recognizable target growth and add state pool training.
2. Evaluate persistence beyond training rollout lengths.
3. Train and measure recovery after damage.
4. Run controlled robustness experiments.
5. Build an interactive playground around validated models.

Generated runs, checkpoints, and exports belong under `data/` and are excluded from Git.
