# Dynamic Monte Carlo training

The retained method assigns either 16 or 32 continuation samples to every
interior prefix of a question. It uses current-policy gradient measurements to
spend additional samples where estimated MC noise most affects other training
questions. It shares the ordinary on-policy trainer with uniform MC24/32 and
outcome-credit baselines.

## Allocation rule

For each update:

1. Sample 8 roots for each of 16 source questions. Interior prefix positions are
   32, 64, 128, 256, 512, 1024, and 2048 tokens, whenever below the root length.
   The response cap is 3072 tokens.
2. Sample 16 continuations per prefix. Split these samples into four groups and
   form three Walsh contrasts for each source question. These contrasts cancel
   root and terminal terms while measuring MC noise.
3. Sample two fresh roots from each of the other 1008 training questions. Split
   those questions into fixed halves and capture their outcome-credit gradients.
   Test questions are excluded from this reference population.
4. Capture the pilot gradient and per-question contrasts without updating any
   weights. Project the contrasts through the derivative of the normalized SGD
   update onto both reference gradients. Average the six squared projections
   to obtain each question's risk score.
5. Rank by risk divided by the number of interior prefixes. Greedily top up
   whole questions from MC16 to MC32 while total suffix calls remain at most
   `24 × number_of_prefixes`. Ties use question ID; questions that do not fit
   are skipped. Freeze the decision before sampling the top-up.
6. Reuse the pilot samples, assemble adjacent-value process credit, and take one
   ordinary SGD step. The selector restores scheduler and RNG state first.

For pilot gradient `g`, contrast `z`, reference gradient `h`, and nominal step
`δ = 0.01`, the squared projection is
`[δ / ||g|| × (z·h − (g·h)(z·g)/||g||²)]²`.
The implementation and deterministic allocator are in
[`math_mc_cross_risk.py`](../src/rllab/math_mc_cross_risk.py).

Root baselines are leave-one-out outcomes from the same question. Interior
values are MC means; terminal values are observed outcomes. Each token block
receives the difference of its adjacent values. The loss sums over tokens and
averages over root sequences, preserving the full global denominator even when
zero-credit computation is omitted. No learned process reward model is used.

## Inputs and frozen plans

Provide the pinned Llama-3.1-8B-Instruct snapshot (revision
`0e9e39f249a16976918f6564b8830bc894c89659`) and `questions.jsonl`.
The repository includes the [fixed question identities](../configs/rl/math_questions.json)
from the final comparison: 1024 training and 512 test questions, with the
historical duplicate-family IDs and split assignments. No problem text or
answers are included. Download the two PRM800K files named in that manifest
using its pinned URLs, saving them as `math_prm_split_train.jsonl` and
`math_prm_split_test.jsonl`, then reconstruct the selected dataset:

```bash
python scripts/prepare_math_data.py \
  --source-dir /path/to/prm800k-files --output /path/to/math-data
```

The helper verifies both source hashes and preserves the original question
records. Its output contains only the selected questions, so its dataset hash
and candidate-pool counts differ from the historical combined MATH/GSM/DAPO
file; the selected rows, prompts and training schedule are reproducible.
PRM800K source-file membership is not the study's train/test assignment: use the
checked-in manifest's assignments, which account for duplicate families.

For a custom dataset, each JSONL row needs `question_id`, `group_id`, `source`,
`split`, `level`, `problem`, and `answer`. MATH rows use `source: "math"`,
`split: "train"` or `"test"`, and integer `level: 5`. `answer` is the final
answer for the symbolic checker. No duplicate family may cross the split.
Preparation filters unsupported text and prompts above 1024 tokens, and freezes
selected IDs, prompts, schedule, seed, source hash, and controls in `plan.json`.
Different question identities or content define a different experiment.

```bash
python scripts/prepare_math_training.py \
  --profile configs/rl/llama_math.yaml \
  --model /path/to/Llama-3.1-8B-Instruct \
  --dataset /path/to/math-data/questions.jsonl \
  --allocation dynamic --devices 8 \
  --output /path/to/plans/dynamic
```

Prepare `uniform24` and `uniform32` using the same inputs and changing only
`--allocation` and `--output`. Selection and random keys are deterministic.
All three public recipes start from the original model for 64 updates, use
128 global roots, temperature 0.6, nominal step norm 0.01, evaluate every 8
updates, and checkpoint every 4 updates. Main evaluations use seed 20260922;
a separate final test evaluation uses seed 20260914. Each evaluation estimates
sampled pass@1 from four answers per question. Training and test cohorts are
both recorded; analyze them separately instead of using their combined mean.

For ORM versus fixed-MC PRM, omit `--allocation` and use one frozen base plan
with `--credit orm` and `--credit prm`. The Qwen base profile remains available
for the common plan and credit checks. Dynamic allocation currently enforces
the Llama study's population and controls.

## Launch and resume

Use the [native environment](installation.md), commit the checkout, and submit
one node with eight devices through the Task entry:

```bash
export RLLAB_STORAGE_ROOT=/path/to/rllab-storage
export RLLAB_TASK_MODE=math_train
export RLLAB_MATH_PLAN=/path/to/plans/dynamic/plan.json
export RLLAB_CREDIT=prm
export RLLAB_RUN_TAG=llama-math-dynamic
export RLLAB_NPROC_PER_NODE=8
bash scripts/run_task.sh
```

Each arm has a driver lock, launch/utilization logs, immutable copied plan,
graded samples, scalar selection receipts, committed checkpoints, and
`state.json` under `$RLLAB_STORAGE_ROOT/task-logs/$RLLAB_RUN_TAG/prm/`.
Use distinct tags for the uniform conditions. `RLLAB_RESUME=1` restores the
last committed checkpoint with the same plan, credit mode, and topology.
Completed arms are not restarted. `RLLAB_STOP_AFTER` and `RLLAB_TIME_LIMIT`
bound a development run without changing the plan. Formal plans have a
36-hour cumulative wall budget. Only the latest two committed checkpoints are
kept; original samples and receipts are retained.

Optional `RLLAB_ROLLOUT_MAX_NUM_SEQS` and
`RLLAB_ROLLOUT_GPU_MEMORY_UTILIZATION` alter execution capacity, not the global
scientific batch or MC rule. Resolved settings are recorded for each attempt.

## Analysis and reproduction scope

```bash
python scripts/analyze_math_training.py \
  --run-dir /path/to/rllab-storage/task-logs/llama-math-dynamic \
  --credit prm --regrade-mc --output /path/to/dynamic-analysis.json
```

The analyzer independently reconstructs credit, suffix counts, selection order,
reference exclusions, gradient receipts, update scale, and checkpoint lineage.
It records per-question curves and counts reference-generation and selector
costs. Check its `--help` for paired ORM/PRM analysis. Root/target evaluation
answers are regraded; `--regrade-mc` also regrades all suffix answers.

The historical dynamic run completed 24 updates and then continued to 64 using
16 devices. The uniform MC24 comparator used 8 devices. The checkpoint lineage
reader in `math_mc_cross_risk_continuation.py` preserves that explicitly bound
resume format. New public recipes start at step zero on 8 devices; they
reproduce the method and controls, not bitwise historical trajectories.

The historical one-seed comparison supports a limited endpoint observation.
Adaptive sample reuse and the topology difference limit causal interpretation.
The MC24 ceiling counts suffix calls only: reference roots and up to 81 gradient
captures per update are additional work. Report their token and wall costs
before making any compute-efficiency claim.
