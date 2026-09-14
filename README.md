# ncap

Neural Cellular Automata Playground. A PyTorch project that learns to grow images and recover from damage.

Includes training, simulation, and recovery experiments. The interactive playground is planned. Leaf and butterfly experiments show early growth and partial recovery. Long-term stability still needs work.

## Setup

Requires Python 3.11+ and a compatible PyTorch build.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Train and simulate

Train the included leaf target, then export a rollout:

```sh
ncap-train --target assets/targets/leaf.png --config configs/leaf-small.json \
  --output data/runs/leaf

ncap-simulate --checkpoint data/runs/leaf/checkpoint.pt \
  --steps 80 --output data/exports/leaf
```

Outputs include a checkpoint, loss metrics, a final image, and a growth GIF. Use a new output directory for each run. Generated files stay under `data/` and are ignored by Git.

An original butterfly target is also included at `assets/targets/butterfly.png`. Target generators are in `scripts/`.

## Experiments

Use `configs/regeneration.json` for damage training, or run the fixed recovery comparison:

```sh
ncap-study --target assets/targets/leaf.png --plan configs/recovery-study.json \
  --output data/studies/leaf-recovery
```

Compare long-term growth using the models from that study:

```sh
ncap-persistence --study data/studies/leaf-recovery \
  --plan configs/persistence-study.json --output data/studies/leaf-persistence
```

Other experiments:

| Command | Purpose |
| --- | --- |
| `ncap-evaluate` | Check one model |
| `ncap-recovery` | Test damage recovery |
| `ncap-horizons` | Compare training rollout lengths |
| `ncap-bounds` | Test cell-state limits |
| `ncap-objective` | Compare training penalties |
| `ncap-budgets` | Compare training budgets and recovery |

Plans are in `configs/`. The larger budget comparison uses `configs/extended-budget-study.json`; `--workers 3` runs its three training seeds in parallel. Run any command with `--help` for options.

## Tests

```sh
python -m pytest
```
