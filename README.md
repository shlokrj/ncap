# ncap

Neural Cellular Automata Playground. A PyTorch project that learns to grow images and recover from damage.

Includes training, simulation, and recovery experiments. The interactive playground is planned. Leaf experiments show early growth and partial recovery. Long-term stability still needs work.

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

Compare long-term growth using the models from that study:

```sh
ncap-persistence --study data/studies/leaf-recovery \
  --plan configs/persistence-study.json --output data/studies/leaf-persistence
```

`ncap-evaluate` checks one model and `ncap-recovery` tests recovery after damage. `ncap-horizons` compares short and long training rollouts using `configs/horizon-study.json`. Run any command with `--help` for options.

## Tests

```sh
python -m pytest
```
