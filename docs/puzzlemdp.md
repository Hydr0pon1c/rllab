# PuzzleMDP

PuzzleMDP is a deterministic layered DAG. Each nonterminal state has a fixed
number of actions; each action incurs a cost and moves to the next layer.
Entering a terminal state also incurs its terminal cost. Rewards are negative
costs. Dynamic programming computes exact values, Q-values, advantages,
occupancy, and derivatives with respect to policy logits.

## Environment API

[examples/puzzlemdp.py](../examples/puzzlemdp.py) demonstrates generation, exact
policy evaluation, counterfactual transitions, and snapshot/restore. The public
API is in `rllab.mdp`; `rllab.mdp.solve` also provides optimal policies and the
performance-difference identity. Generation and evaluation use explicit seeds
and do not require Torch.

```bash
rllab-data task probe --config configs/data/smoke.yaml --count 2
rllab-data task probe --config configs/data/study-v2-h32.yaml --count 2
```

The smoke configuration is a tiny development puzzle. The H32 configuration
preserves the study's horizon, branching, cost distribution, and difficulty
filters. Changing these factors defines a different environment.

## Tokenized datasets

Install the tokenizer extra and fetch the pinned Qwen3-8B snapshot. Model
registration records its source and revision.

```bash
pip install -e '.[tokenizer,geometry]'
export RLLAB_STORAGE_ROOT=/path/to/rllab-storage
rllab-data storage init
rllab-data model fetch
rllab-data dataset build --config configs/data/smoke.yaml
rllab-data dataset build --config configs/data/study-v2-h32.yaml
rllab-data dataset list
```

Builds are immutable and expose a `latest` link. Use the printed version path
in a frozen experiment configuration; verify it with
`rllab-data dataset verify PATH`. The state/action token registry enforces
single-token IDs, disjoint token roles, and fixed prompt layouts.

## LLM training

Use a CUDA-compatible Torch environment with the model dependencies installed;
see [installation](installation.md). The checked-in ORM and exact-PRM recipes
use the same MDP population, trajectory budget, initialization, and update scale.
Their `data.path` initially points at `latest`; replace it with your immutable
build path before freezing a run. Set `output.root` to external storage.

```bash
export RLLAB_TASK_MODE=puzzlemdp
export RLLAB_NPROC_PER_NODE=8
export RLLAB_RUN_TAG=puzzlemdp-prm
export RLLAB_PUZZLE_CONFIG=configs/experiment/puzzlemdp-prm.yaml
bash scripts/run_task.sh
```

Use `puzzlemdp-orm.yaml` and a different run tag for the paired ORM condition.
For local debugging, `rllab-exp train --config PATH --allow-dirty` exposes the
same trainer. Formal Task launches require committed source. No training is
performed by the CPU examples or unit tests.

`rllab-exp geometry --config PATH --name NAME` builds measured aligned/adversarial
relabeling overlays. Set `data.geometry_path` to the overlay and
`data.labeling_mode` to the matching mode in a copied experiment configuration.
Geometry construction requires model gradients and is substantially more
expensive than the pure environment example.

## Scientific controls

Rollout, loss, and evaluation share one masked action softmax. Fixed initial
logit offsets make each state's initial policy uniform. The trainer uses
full-parameter SGD without momentum, clipping, whitening, entropy bonuses,
weight decay, or KL penalties. Exact PuzzleMDP evaluation is distinct from
sampled math answer evaluation.

Exposure distinguishes states that receive a policy loss, states whose IDs
appear only as destinations, and states whose IDs never appear in the update
prompts. Preserve these denominators when interpreting cross-state effects.
Relabeling changes the token assignment while preserving the abstract MDP.
