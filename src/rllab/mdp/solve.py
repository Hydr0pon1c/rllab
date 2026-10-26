"""Exact finite-horizon policy analysis for deterministic PuzzleMDPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP, validate_mdp

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class MDPArrays:
    state_ids: tuple[str, ...]
    state_index: dict[str, int]
    depths: IntArray
    terminal: NDArray[np.bool_]
    terminal_costs: FloatArray
    targets: IntArray
    costs: FloatArray


@dataclass(frozen=True)
class PolicyEvaluation:
    """Exact undiscounted return quantities under one policy.

    Rewards are negative transition costs. Terminal values are zero. Q/A values
    on terminal rows are NaN because there is no valid action there.
    """

    policy: FloatArray
    value: FloatArray
    q_value: FloatArray
    advantage: FloatArray
    occupancy: FloatArray
    logit_gradient: FloatArray
    start_state: str
    objective: float


@dataclass(frozen=True)
class PerformanceDifference:
    starts: tuple[str, ...]
    exact_delta_j: FloatArray
    advantage_projection: FloatArray
    residual: FloatArray


def mdp_arrays(mdp: PuzzleMDP) -> MDPArrays:
    validate_mdp(mdp)
    states = mdp.ordered_states()
    state_ids = tuple(state.state_id for state in states)
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    n_states = len(states)
    targets = np.full((n_states, mdp.num_actions), -1, dtype=np.int64)
    costs = np.full((n_states, mdp.num_actions), np.nan, dtype=np.float64)
    for transition in mdp.transitions:
        source = state_index[transition.source]
        action = transition.action
        targets[source, action] = state_index[transition.target]
        costs[source, action] = transition.cost
    return MDPArrays(
        state_ids=state_ids,
        state_index=state_index,
        depths=np.asarray([state.depth for state in states], dtype=np.int64),
        terminal=np.asarray([state.terminal for state in states], dtype=np.bool_),
        terminal_costs=np.asarray(
            [state.terminal_cost for state in states], dtype=np.float64
        ),
        targets=targets,
        costs=costs,
    )


def uniform_policy(mdp: PuzzleMDP) -> FloatArray:
    arrays = mdp_arrays(mdp)
    policy = np.zeros_like(arrays.costs)
    policy[~arrays.terminal] = 1.0 / mdp.num_actions
    return policy


def optimal_policy(mdp: PuzzleMDP) -> FloatArray:
    arrays = mdp_arrays(mdp)
    n_states = len(arrays.state_ids)
    policy = np.zeros((n_states, mdp.num_actions), dtype=np.float64)
    value = -arrays.terminal_costs.copy()
    for depth in reversed(range(mdp.horizon)):
        for state in np.flatnonzero(arrays.depths == depth):
            q_values = -arrays.costs[state] + value[arrays.targets[state]]
            best = np.flatnonzero(
                np.isclose(q_values, q_values.max(), atol=1e-12, rtol=0.0)
            )
            policy[state, best] = 1.0 / len(best)
            value[state] = q_values[best[0]]
    return policy


def validate_policy(mdp: PuzzleMDP, policy: NDArray[np.floating]) -> FloatArray:
    arrays = mdp_arrays(mdp)
    result = np.asarray(policy, dtype=np.float64)
    expected_shape = (len(arrays.state_ids), mdp.num_actions)
    if result.shape != expected_shape:
        raise ValidationError(
            f"policy shape is {result.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValidationError("policy entries must be finite and non-negative")
    nonterminal_sums = result[~arrays.terminal].sum(axis=1)
    if not np.allclose(nonterminal_sums, 1.0, atol=1e-10, rtol=0.0):
        raise ValidationError("every non-terminal policy row must sum to one")
    if not np.allclose(result[arrays.terminal], 0.0, atol=1e-12, rtol=0.0):
        raise ValidationError("terminal policy rows must be all zero")
    return result


def occupancy_from(
    mdp: PuzzleMDP,
    policy: NDArray[np.floating],
    start_state: str | None = None,
) -> FloatArray:
    arrays = mdp_arrays(mdp)
    checked_policy = validate_policy(mdp, policy)
    resolved_start = start_state or mdp.start_state
    if resolved_start not in arrays.state_index:
        raise ValidationError(f"unknown occupancy start state: {resolved_start}")
    occupancy = np.zeros(len(arrays.state_ids), dtype=np.float64)
    occupancy[arrays.state_index[resolved_start]] = 1.0
    start_depth = int(arrays.depths[arrays.state_index[resolved_start]])
    for depth in range(start_depth, mdp.horizon):
        for state in np.flatnonzero(arrays.depths == depth):
            if occupancy[state] == 0.0:
                continue
            np.add.at(
                occupancy,
                arrays.targets[state],
                occupancy[state] * checked_policy[state],
            )
    return occupancy


def evaluate_policy(
    mdp: PuzzleMDP,
    policy: NDArray[np.floating],
    *,
    start_state: str | None = None,
) -> PolicyEvaluation:
    arrays = mdp_arrays(mdp)
    checked_policy = validate_policy(mdp, policy)
    n_states = len(arrays.state_ids)
    value = -arrays.terminal_costs.copy()
    q_value = np.full((n_states, mdp.num_actions), np.nan, dtype=np.float64)
    advantage = np.full_like(q_value, np.nan)

    for depth in reversed(range(mdp.horizon)):
        for state in np.flatnonzero(arrays.depths == depth):
            q_value[state] = -arrays.costs[state] + value[arrays.targets[state]]
            value[state] = float(np.dot(checked_policy[state], q_value[state]))
            advantage[state] = q_value[state] - value[state]

    resolved_start = start_state or mdp.start_state
    occupancy = occupancy_from(mdp, checked_policy, resolved_start)
    logit_gradient = np.zeros_like(q_value)
    nonterminal = ~arrays.terminal
    logit_gradient[nonterminal] = (
        occupancy[nonterminal, None]
        * checked_policy[nonterminal]
        * advantage[nonterminal]
    )
    return PolicyEvaluation(
        policy=checked_policy,
        value=value,
        q_value=q_value,
        advantage=advantage,
        occupancy=occupancy,
        logit_gradient=logit_gradient,
        start_state=resolved_start,
        objective=float(value[arrays.state_index[resolved_start]]),
    )


def all_start_occupancies(mdp: PuzzleMDP, policy: NDArray[np.floating]) -> FloatArray:
    arrays = mdp_arrays(mdp)
    checked_policy = validate_policy(mdp, policy)
    return np.stack(
        [
            occupancy_from(mdp, checked_policy, state_id)
            for state_id in arrays.state_ids
        ],
        axis=0,
    )


def performance_difference(
    mdp: PuzzleMDP,
    base_policy: NDArray[np.floating],
    updated_policy: NDArray[np.floating],
    *,
    starts: Iterable[str] | None = None,
) -> PerformanceDifference:
    """Evaluate the exact performance-difference identity from every start.

    For this undiscounted DAG, ``d`` is the unnormalised state visitation
    probability. The projection uses ``sum d^{pi'} (pi' - pi) A^pi``.
    """

    arrays = mdp_arrays(mdp)
    base = validate_policy(mdp, base_policy)
    updated = validate_policy(mdp, updated_policy)
    base_eval = evaluate_policy(mdp, base)
    updated_eval = evaluate_policy(mdp, updated)
    start_ids = tuple(starts) if starts is not None else arrays.state_ids
    unknown = set(start_ids) - set(arrays.state_ids)
    if unknown:
        raise ValidationError(
            f"unknown performance-difference starts: {sorted(unknown)}"
        )

    exact = np.asarray(
        [
            updated_eval.value[arrays.state_index[state_id]]
            - base_eval.value[arrays.state_index[state_id]]
            for state_id in start_ids
        ],
        dtype=np.float64,
    )
    projections: list[float] = []
    for start_state in start_ids:
        occupancy = occupancy_from(mdp, updated, start_state)
        per_state = np.nansum((updated - base) * base_eval.advantage, axis=1)
        projections.append(float(np.dot(occupancy, per_state)))
    projected = np.asarray(projections, dtype=np.float64)
    return PerformanceDifference(
        starts=start_ids,
        exact_delta_j=exact,
        advantage_projection=projected,
        residual=exact - projected,
    )
