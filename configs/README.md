# Configurations

| File | Purpose |
| --- | --- |
| `data/smoke.yaml` | Tiny PuzzleMDP generation and tokenization example |
| `data/study-v2-h32.yaml` | Study H32 generation and difficulty controls |
| `experiment/puzzlemdp-orm.yaml` | Full-parameter outcome-credit PuzzleMDP training |
| `experiment/puzzlemdp-prm.yaml` | Matched exact-process-credit PuzzleMDP training |
| `rl/qwen_math.yaml` | Common math scientific controls and Qwen base profile |
| `rl/math_questions.json` | Fixed question IDs, group assignments, and pinned MATH sources |
| `rl/llama_math.yaml` | Llama actor/model/temperature overrides |
| `rl/math_native.yaml` | Native sampler/engine capacity settings |
| `rl/sgd_base.yaml` | Explicit full-parameter SGD and supplied-credit loss |

Use `scripts/prepare_math_training.py --allocation dynamic`, `uniform24`, or
`uniform32` with the Llama profile to freeze a 64-update plan. The YAML files are
preparation inputs; changing them does not alter a saved plan. See
[training](../docs/training.md) for the full commands and paired controls.

PuzzleMDP examples point at the built dataset's `latest` link. Replace it with
the immutable version path and choose an external output directory before
freezing a formal run.
