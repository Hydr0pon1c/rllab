# RLLab

PuzzleMDP and dynamic Monte Carlo allocation for studying credit assignment and
cross-state generalization in language-model reinforcement learning.

This repository contains two parts:

- **PuzzleMDP:** deterministic, finite-horizon puzzles with exact values,
  advantages, occupancy, policy gradients, and reversible environment state.
  The LLM interface preserves prompt/token invariants and supports isomorphic
  state relabeling, exact evaluation, and paired outcome/process-credit training.
- **Dynamic MC training:** online per-question MC16/32 allocation for MATH,
  with uniform MC24/32 and outcome-credit baselines. The trainer uses
  full-parameter, momentum-free SGD through verl/vLLM.

## Quick start: PuzzleMDP on CPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
rllab-data task probe --config configs/data/smoke.yaml --count 2
python examples/puzzlemdp.py
pytest -q tests/test_mdp.py tests/test_config.py
```

The environment and exact solver need NumPy and PyYAML; no model download or
accelerator is needed for this example.

## Documentation

| Task | Guide |
| --- | --- |
| Generate puzzles, build tokenized datasets, train an LLM | [PuzzleMDP](docs/puzzlemdp.md) |
| Prepare paired plans, run dynamic MC training, reconstruct results | [Training](docs/training.md) |
| Install the native training stack and select storage paths | [Installation](docs/installation.md) |
| Code map and tests | [Development](docs/development.md) |
| Configuration entry points | [Configs](configs/README.md) |

Models, datasets, checkpoints and logs are external to Git. `RLLAB_STORAGE_ROOT`
selects storage (default `~/.cache/rllab`); `rllab-data storage init` creates
local ignored `data` and `models` links. Python training commands expect an
editable checkout so the checked-in YAML configurations remain available.

The dynamic allocator bounds suffix calls at the cost of uniform MC24 on the
same sampled prefixes. Reference generation and gradient measurements add work;
this is not a claim of reduced end-to-end compute. The original endpoint
comparison used one seed and different device counts across conditions; see
[reproduction scope](docs/training.md#analysis-and-reproduction-scope).

Third-party grading code retains its [license and provenance](THIRD_PARTY.md).
