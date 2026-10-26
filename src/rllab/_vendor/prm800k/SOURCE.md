# Upstream provenance

`grader.py`, `math_normalize.py` and `LICENSE` come from
<https://github.com/openai/prm800k/tree/7ecc794703b2877f63226f2477a49b34f9b25163>.
The Python files are under `prm800k/grading/` upstream; the license is at the root.
Downloaded 2026-09-13 for the math MC debugging contract.

The import `from grading import math_normalize` is changed to the relative
`from . import math_normalize`; a trailing blank line is removed. Grading
behavior is unchanged. This checker is
conservative and imperfect, as documented by its authors. It uses SymPy and
`pylatexenc`; it is not an oracle for intermediate reasoning correctness.

Upstream `grader.py` SHA-256 before the import adjustment:
`9e8bbb6f504ee0d8068e1eca031d78174f1c925cfec391996fdf45c21bb016a9`.

Local execution amendment, 2026-10-11: `are_equal_under_sympy` returns false
before general simplification when the parsed difference has `is_zero is False`.
Unknown zero-ness still follows the original simplifier, and all normalization,
tuple and strict-fraction/integer checks remain. The triggering unequal answer
contains `(2004 + sqrt(2))!`; parsing/nonzero inference finishes quickly whereas
general simplification stalls. This is a SymPy nonzero predicate, not a new
numeric closeness tolerance. Native online grading additionally runs in a
separate bounded process; timeout/crash is an execution error, never a reward.
The ledger records full saved-answer replay and legacy-label comparisons.
