# Installation

## CPU environment

Python 3.10 or newer is required. Python 3.12 was used for the recorded CPU
environment (`requirements/data-py312.lock`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[tokenizer,geometry,math-probe,dev]'
```

The core environment only needs NumPy/PyYAML. `tokenizer` adds Hugging Face
snapshot and prompt support; `geometry` adds SciPy; `math-probe` adds the symbolic
answer checker. None of these extras installs or replaces Torch.

## PuzzleMDP model runtime

In a CUDA environment with a compatible driver, install the model runtime via
`pip install -e '.[puzzle,tokenizer]'`. On vendor PPU images use the supplied
Torch and install the remaining dependencies without replacing it. Model
forward/backward, distributed NCCL, and memory capacity depend on the actual
device environment. The reference H32 recipes require at least 80 GiB free
memory per device; this is an execution check, not a scientific threshold.

## Native math training

The math trainer uses internal APIs of **verl v0.9.0**, revision
`483b8a009ba3a97563edee3a19887e4862b8094a`. The recorded environment is in
[requirements/real-rl-ppu.json](../requirements/real-rl-ppu.json): Torch 2.10.0,
vLLM 0.19.0, Transformers 5.6.2, plus vendor PPU builds. Use a compatible
preconfigured verl/vLLM environment and install this checkout with
`python -m pip install --no-deps -e .`; add the CPU extras if absent. The
manifest is provenance, not a portable pip requirements file.

Activate that environment before invoking the scripts, or set
`RLLAB_RL_ENV_ROOT` to a directory containing `venv/bin/python`.
`RLLAB_PYTHON` can instead name a Python executable. Do not run a bulk upstream
Torch/vLLM upgrade over the PPU environment. This release does not claim that an
unmodified upstream CUDA stack has passed device validation.

The reference dynamic allocator uses one node with eight devices and at least
320 GiB host RAM for full-parameter gradient shards. The global batch stays at
16 questions × 8 roots when changing topology. Historical 16-device plans
remain readable; sharded checkpoint resume requires the original topology.

Before reserving devices, run the native CPU tests described in
[development](development.md) and validate a prepared configuration with
`python -m rllab.math_train --plan PLAN --credit prm --run-dir OUTPUT
--devices 8 --check-config`. Configuration validation does not perform sampling
or optimizer updates. CPU tests do not validate device throughput or checkpoint
transfer.

## Storage

Set `RLLAB_STORAGE_ROOT` to a disk outside the checkout, then run
`rllab-data storage init`. It creates `datasets`, `models`, `runs`, `task-logs`,
and related storage directories, plus ignored local `data`/`models` links.
Existing links are never overwritten. Python defaults to `~/.cache/rllab`;
set the environment variable explicitly for a different location.

The examples expect local model snapshots and a normalized MATH JSONL file.
Weights and datasets are not included. Their upstream terms apply separately.
