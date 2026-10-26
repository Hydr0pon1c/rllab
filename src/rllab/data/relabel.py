"""Paired state relabelings over one immutable canonical MDP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from rllab.data.tokens import TokenRegistry, TokenSpec
from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP

LABELING_SCHEMA_VERSION = "state-labeling/v1"


@dataclass(frozen=True)
class StateLabeling:
    labeling_id: str
    mdp_id: str
    mode: str
    seed: int
    registry_fingerprint: str
    state_tokens: Mapping[str, TokenSpec]
    action_tokens: tuple[TokenSpec, ...]
    assignment_score: float | None = None
    assignment_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LABELING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "labeling_id": self.labeling_id,
            "mdp_id": self.mdp_id,
            "mode": self.mode,
            "seed": self.seed,
            "registry_fingerprint": self.registry_fingerprint,
            "state_tokens": {
                state_id: token.to_dict()
                for state_id, token in sorted(self.state_tokens.items())
            },
            "action_tokens": [asdict(token) for token in self.action_tokens],
            "assignment_score": self.assignment_score,
            "assignment_metadata": dict(self.assignment_metadata),
        }


def make_labeling(
    mdp: PuzzleMDP,
    registry: TokenRegistry,
    *,
    mode: str,
    seed: int,
    labeling_id: str | None = None,
) -> StateLabeling:
    """Create identity/random mappings using the exact same token subset."""

    state_ids = tuple(state.state_id for state in mdp.ordered_states())
    if len(registry.action_tokens) != mdp.num_actions:
        raise ValidationError(
            f"registry has {len(registry.action_tokens)} actions, MDP has {mdp.num_actions}"
        )
    if len(state_ids) > len(registry.state_tokens):
        raise ValidationError(
            f"MDP needs {len(state_ids)} state tokens, registry has {len(registry.state_tokens)}"
        )
    tokens = list(registry.state_tokens[: len(state_ids)])
    if mode == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(tokens)
    elif mode != "identity":
        raise ValidationError(f"unsupported direct labeling mode: {mode!r}")
    mapping = dict(zip(state_ids, tokens, strict=True))
    return StateLabeling(
        labeling_id=labeling_id or f"{mdp.mdp_id}-{mode}-s{seed}",
        mdp_id=mdp.mdp_id,
        mode=mode,
        seed=seed,
        registry_fingerprint=registry.fingerprint,
        state_tokens=mapping,
        action_tokens=registry.action_tokens,
        assignment_metadata={
            "method": "seeded-permutation" if mode == "random" else "pool-order"
        },
    )


def make_score_aligned_labeling(
    mdp: PuzzleMDP,
    registry: TokenRegistry,
    state_token_scores: NDArray[np.floating],
    *,
    mode: str,
    seed: int,
    labeling_id: str | None = None,
    score_provenance: Mapping[str, Any] | None = None,
    assignment_blocks: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
) -> StateLabeling:
    """Solve a measured state-token assignment exactly with Hungarian matching.

    ``state_token_scores[s, t]`` must come from a frozen-model probe (for
    example, the predicted policy-update alignment for assigning token ``t`` to
    state ``s``). ``aligned`` maximises and ``adversarial`` minimises the total
    score. No language-model proxy is fabricated in the data builder.
    """

    if mode not in {"aligned", "adversarial"}:
        raise ValidationError("score-based mode must be 'aligned' or 'adversarial'")
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ValidationError(
            "scipy is required for geometry assignments; install rllab-puzzle[geometry]"
        ) from exc

    state_ids = tuple(state.state_id for state in mdp.ordered_states())
    if len(registry.action_tokens) != mdp.num_actions:
        raise ValidationError(
            f"registry has {len(registry.action_tokens)} actions, MDP has {mdp.num_actions}"
        )
    tokens = registry.state_tokens[: len(state_ids)]
    scores = np.asarray(state_token_scores, dtype=np.float64)
    expected_shape = (len(state_ids), len(tokens))
    if scores.shape != expected_shape:
        raise ValidationError(
            f"score matrix is {scores.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValidationError("state-token score matrix contains non-finite values")
    if assignment_blocks is None:
        blocks = ((tuple(range(len(state_ids))), tuple(range(len(tokens)))),)
        assignment_constraint = "global"
    else:
        blocks = tuple(
            (
                tuple(int(index) for index in row_indices),
                tuple(int(index) for index in column_indices),
            )
            for row_indices, column_indices in assignment_blocks
        )
        assignment_constraint = "explicit_blocks"
    if not blocks or any(
        not row_indices or len(row_indices) != len(column_indices)
        for row_indices, column_indices in blocks
    ):
        raise ValidationError(
            "assignment blocks must be non-empty square row/column partitions"
        )
    flat_rows = [index for row_indices, _ in blocks for index in row_indices]
    flat_columns = [index for _, column_indices in blocks for index in column_indices]
    expected_indices = list(range(len(state_ids)))
    if (
        sorted(flat_rows) != expected_indices
        or sorted(flat_columns) != expected_indices
    ):
        raise ValidationError(
            "assignment blocks must partition every state row and token column exactly once"
        )

    assignments: list[tuple[int, int]] = []
    for row_indices, column_indices in blocks:
        block_scores = scores[np.ix_(row_indices, column_indices)]
        local_rows, local_columns = linear_sum_assignment(
            block_scores, maximize=mode == "aligned"
        )
        if not np.array_equal(local_rows, np.arange(len(row_indices))):
            raise AssertionError("unexpected incomplete Hungarian block assignment")
        assignments.extend(
            (row_indices[int(local_row)], column_indices[int(local_column)])
            for local_row, local_column in zip(local_rows, local_columns, strict=True)
        )
    assignments.sort()
    rows = np.asarray([row for row, _ in assignments], dtype=np.int64)
    columns = np.asarray([column for _, column in assignments], dtype=np.int64)
    if not np.array_equal(rows, np.arange(len(state_ids))):
        raise AssertionError("unexpected incomplete Hungarian assignment")
    mapping = {
        state_ids[row]: tokens[column]
        for row, column in zip(rows, columns, strict=True)
    }
    score = float(scores[rows, columns].sum())
    score_hash = sha256(scores.astype("<f8", copy=False).tobytes(order="C")).hexdigest()
    return StateLabeling(
        labeling_id=labeling_id or f"{mdp.mdp_id}-{mode}-s{seed}",
        mdp_id=mdp.mdp_id,
        mode=mode,
        seed=seed,
        registry_fingerprint=registry.fingerprint,
        state_tokens=mapping,
        action_tokens=registry.action_tokens,
        assignment_score=score,
        assignment_metadata={
            "method": "scipy-linear-sum-assignment",
            "objective": "maximize" if mode == "aligned" else "minimize",
            "assignment_constraint": assignment_constraint,
            "assignment_block_sizes": [len(rows) for rows, _ in blocks],
            "score_matrix_sha256": score_hash,
            "score_matrix_shape": list(scores.shape),
            "provenance": dict(score_provenance or {}),
        },
    )
