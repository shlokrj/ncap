# ncap

Neural Cellular Automata Playground — exploring how a shared local neural rule can learn growth, persistence, and regeneration.

## Status

The PyTorch core, optional state-pool training, and persistence diagnostics are implemented: load an RGBA target, train a shared update rule, save a checkpoint, and export a seeded rollout. Regeneration, controlled experiments, and the playground are next. A small geometric-leaf trial produces recognizable early growth but drifts over longer rollouts. Long-term stability and regeneration have not been demonstrated.

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

Supply a local image or use the bundled original geometric leaf (`assets/targets/leaf.png`, reproducible with `python scripts/make_leaf.py`). No pretrained checkpoints are bundled. Images are fitted into the configured square grid with aspect ratio preserved and transparent padding. The objective is mean squared error over premultiplied RGB and alpha, so invisible RGB values do not affect the loss. Exports composite the visible channels on white.

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

Training runs on CPU. The base configuration starts every batch from fresh seeds. Set `pool_size` to at least `batch_size` to reuse grown states; zero disables the pool. Each sampled batch replaces its highest-loss state with a fresh seed, and rollout results return to the pool without their gradient history. Checkpoints also retain pool contents and the sampling RNG state. The final metric uses the training target and one evaluation seed; it is a diagnostic, not held-out evidence or a stability test. Checkpoints support simulation; a resume-training command is not implemented. Exact replay is tested within the same runtime, not promised across platforms or PyTorch versions.

## State pools and persistence

```sh
ncap-train --target assets/targets/leaf.png --config configs/leaf-small.json \
  --output data/runs/leaf-small
ncap-evaluate --checkpoint data/runs/leaf-small/checkpoint.pt \
  --target data/runs/leaf-small/target-source --horizons 40 80 160 320 \
  --seeds 10000 10001 10002 --output data/exports/leaf-persistence
```

`configs/leaf-small.json` is a small 24×24, 1,000-iteration exploratory run. `configs/growth.json` enables a 128-state pool with the larger base settings. Neither is a validated convergence recipe.

Evaluation follows each seeded trajectory continuously through the requested horizons; it does not restart at each horizon. It records visible-channel MSE, alpha-mask intersection over union (threshold 0.1), foreground cell count, and a PNG at each horizon. The output includes checkpoint and target hashes, seeds, settings, and runtime versions. It refuses existing output folders and records failures. Compare later horizons with the training horizon to investigate decay or divergence. These are same-target diagnostics, not held-out evidence or a guarantee of stability.

### Exploratory leaf result

One 1,000-iteration run with `configs/leaf-small.json` produced a recognizable low-resolution leaf. Across evaluation seeds 10000–10002, alpha IoU was 0.797–0.874 at step 40 and fell to 0.613–0.678 at step 320. This is evidence of early growth and later drift on one training seed and one target; it does not establish robustness or a benefit over seed-only training. Run artifacts remain local under ignored `data/`.

## Next milestones

1. Validate target growth across training seeds and targets.
2. Evaluate persistence beyond training rollout lengths.
3. Train and measure recovery after damage.
4. Run controlled robustness experiments.
5. Build an interactive playground around validated models.

Generated runs, checkpoints, and exports belong under `data/` and are excluded from Git.
