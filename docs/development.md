# Development

## Code map

| Code | Responsibility |
| --- | --- |
| `src/rllab/mdp/` | Deterministic environment, generation, exact solvers |
| `src/rllab/data/` | Token/prompt invariants, credit, exposure, relabeling |
| `src/rllab/experiment/` | LLM policy, exact evaluation, geometry, SGD training |
| `config.py`, `artifacts.py`, `storage.py`, `models.py` | Immutable dataset builds and local storage |
| `math_credit.py`, `math_grading.py` | MC/block credit and isolated answer grading |
| `math_training.py`, `real_math_loop.py` | Frozen plans, sampling jobs, credit assembly, resume rules |
| `math_mc_cross_risk*.py` | Risk, allocation, online selection, independent receipt audit |
| `math_alignment*.py` | Frozen gradient capture and correct distributed denominators |
| `math_train.py`, `real_rl_runtime.py` | Native verl/vLLM trainer and audited SGD |
| `scripts/prepare_math_data.py` | Rebuild the selected MATH records from pinned sources |
| `scripts/prepare_math_training.py` | Paired plan preparation |
| `scripts/analyze_math_training.py` | Independent reconstruction of committed training records |

## Checks

CPU checks, after installing the development and math/tokenizer extras:

```bash
python -m pytest -q -m 'not real_rl'
ruff check src scripts tests examples
python examples/puzzlemdp.py
```

Tests requiring local tokenizer snapshots skip when those resources are absent.
The `real_rl` marker selects framework-dependent CPU tests. In the compatible
native environment, run:

```bash
CUDA_VISIBLE_DEVICES='' bash scripts/real_rl_python.sh -m pytest -q -m real_rl
bash -n scripts/run_task.sh scripts/run_math_task.sh scripts/real_rl_python.sh
```

These tests cover signed credit and loss denominators, gradient capture without
parameter writes, RNG/scheduler restoration, sharding weights, and resume at the
final checkpoint. They do not establish accelerator throughput or distributed
checkpoint recovery. Never replace vendor Torch to make a test environment work.

## Changes and experiments

Inspect Git status first and preserve unrelated user changes. Keep models,
datasets and generated artifacts outside Git. Reanalysis writes new files;
original results and predictions are immutable. Submitted runs own their frozen
source and plans.

A new experiment needs a minimal contract describing the question, alternatives,
intervention/controls, metrics, interpretations, budget, stop condition and next
decision. Routine refactoring and tests do not require a new experiment. Preserve
prompt/token invariants, exact evaluation and paired conditions in PuzzleMDP;
preserve explicit loss, sampling and budget controls in math training. Null or
contrary results are valid findings.
