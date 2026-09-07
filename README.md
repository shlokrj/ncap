# ncap

Neural Cellular Automata Playground — a PyTorch project exploring how simple local rules can learn to grow images and recover from damage.

Currently includes training, simulation, and recovery experiments. The interactive playground is planned. Early leaf experiments show growth and partial recovery; long-term stability remains a work in progress.

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

## Experiments

Use `configs/regeneration.json` for damage training, or run the fixed recovery comparison:

```sh
ncap-study --target assets/targets/leaf.png --plan configs/recovery-study.json \
  --output data/studies/leaf-recovery
```

`ncap-evaluate` measures persistence and `ncap-recovery` tests recovery after damage. Run any command with `--help` for options.

## Tests

```sh
python -m pytest
```
