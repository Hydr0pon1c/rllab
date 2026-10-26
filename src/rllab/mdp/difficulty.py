"""Measured task difficulty and constraint-based synthesis.

Difficulty labels are never inferred from horizon alone. Each report is derived
from the exact solved MDP and records structural, exploration, shortcut, and
credit-delay properties separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from math import log
from typing import Any

import numpy as np

from rllab.config import DifficultyConfig, GeneratorConfig
from rllab.errors import ConfigurationError
from rllab.mdp.generate import generate_layered_mdp
from rllab.mdp.schema import PuzzleMDP
from rllab.mdp.solve import evaluate_policy, mdp_arrays, optimal_policy, uniform_policy


@dataclass(frozen=True)
class DifficultyReport:
    num_states: int
    num_nonterminal_states: int
    horizon: int
    branching_factor: int
    layer_widths: tuple[int, ...]
    optimal_start_cost: float
    uniform_start_cost: float
    uniform_regret_per_step: float
    mean_q_gap: float
    min_q_gap: float
    max_q_gap: float
    immediate_greedy_accuracy: float
    delayed_choice_rate: float
    mean_depth_entropy: float
    min_nonzero_uniform_occupancy: float
    mean_reconvergence_steps: float | None
    never_reconverge_rate: float
    terminal_cost_std: float
    optimal_action_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reachable_sets_by_extra_steps(mdp: PuzzleMDP) -> dict[tuple[str, int], set[str]]:
    outgoing = mdp.outgoing()
    result: dict[tuple[str, int], set[str]] = {}
    for state in reversed(mdp.ordered_states()):
        result[(state.state_id, 0)] = {state.state_id}
        for extra in range(1, mdp.horizon - state.depth + 1):
            descendants: set[str] = set()
            for transition in outgoing[state.state_id]:
                descendants.update(result[(transition.target, extra - 1)])
            result[(state.state_id, extra)] = descendants
    return result


def _reconvergence_metrics(mdp: PuzzleMDP) -> tuple[float | None, float]:
    outgoing = mdp.outgoing()
    reachable = _reachable_sets_by_extra_steps(mdp)
    distances: list[int] = []
    never = 0
    total = 0
    for state in mdp.ordered_states():
        if state.terminal:
            continue
        targets = [transition.target for transition in outgoing[state.state_id]]
        remaining = mdp.horizon - state.depth - 1
        for left, right in combinations(targets, 2):
            total += 1
            distance = None
            for extra in range(1, remaining + 1):
                if reachable[(left, extra)] & reachable[(right, extra)]:
                    distance = extra + 1  # include the first counterfactual action
                    break
            if distance is None:
                never += 1
            else:
                distances.append(distance)
    mean_distance = float(np.mean(distances)) if distances else None
    return mean_distance, (never / total if total else 0.0)


def measure_difficulty(mdp: PuzzleMDP) -> DifficultyReport:
    arrays = mdp_arrays(mdp)
    uniform = uniform_policy(mdp)
    optimal = optimal_policy(mdp)
    uniform_eval = evaluate_policy(mdp, uniform)
    optimal_eval = evaluate_policy(mdp, optimal)

    q_cost = -optimal_eval.q_value[~arrays.terminal]
    sorted_q_cost = np.sort(q_cost, axis=1)
    q_gaps = sorted_q_cost[:, 1] - sorted_q_cost[:, 0]
    immediate_costs = arrays.costs[~arrays.terminal]
    optimal_actions = np.argmin(q_cost, axis=1)
    greedy_credit = []
    for row, best_action in zip(immediate_costs, optimal_actions, strict=True):
        cheapest = np.flatnonzero(np.isclose(row, row.min(), atol=0.0, rtol=0.0))
        greedy_credit.append((1.0 / len(cheapest)) if best_action in cheapest else 0.0)
    immediate_accuracy = float(np.mean(greedy_credit))

    depth_entropies: list[float] = []
    for depth in range(1, mdp.horizon + 1):
        indices = np.flatnonzero(arrays.depths == depth)
        probabilities = uniform_eval.occupancy[indices]
        mass = probabilities.sum()
        if mass <= 0 or len(indices) <= 1:
            depth_entropies.append(1.0)
            continue
        normalized = probabilities / mass
        entropy = -float(np.sum(normalized * np.log(normalized + 1e-300)))
        depth_entropies.append(entropy / log(len(indices)))

    start_index = arrays.state_index[mdp.start_state]
    optimal_start_cost = -float(optimal_eval.value[start_index])
    uniform_start_cost = -float(uniform_eval.value[start_index])
    nonzero = uniform_eval.occupancy[uniform_eval.occupancy > 0]
    mean_reconvergence, never_reconverge = _reconvergence_metrics(mdp)
    terminal_costs = arrays.terminal_costs[arrays.terminal]
    action_counts = np.bincount(optimal_actions, minlength=mdp.num_actions)

    return DifficultyReport(
        num_states=len(arrays.state_ids),
        num_nonterminal_states=int((~arrays.terminal).sum()),
        horizon=mdp.horizon,
        branching_factor=mdp.num_actions,
        layer_widths=tuple(
            int(np.sum(arrays.depths == depth)) for depth in range(mdp.horizon + 1)
        ),
        optimal_start_cost=optimal_start_cost,
        uniform_start_cost=uniform_start_cost,
        uniform_regret_per_step=(uniform_start_cost - optimal_start_cost) / mdp.horizon,
        mean_q_gap=float(np.mean(q_gaps)),
        min_q_gap=float(np.min(q_gaps)),
        max_q_gap=float(np.max(q_gaps)),
        immediate_greedy_accuracy=immediate_accuracy,
        delayed_choice_rate=1.0 - immediate_accuracy,
        mean_depth_entropy=float(np.mean(depth_entropies)),
        min_nonzero_uniform_occupancy=float(nonzero.min()),
        mean_reconvergence_steps=mean_reconvergence,
        never_reconverge_rate=never_reconverge,
        terminal_cost_std=float(np.std(terminal_costs)),
        optimal_action_counts=tuple(int(value) for value in action_counts),
    )


def difficulty_violations(
    report: DifficultyReport,
    constraints: DifficultyConfig,
) -> tuple[str, ...]:
    checks = (
        (
            "delayed_choice_rate",
            report.delayed_choice_rate,
            constraints.min_delayed_choice_rate,
            constraints.max_delayed_choice_rate,
        ),
        (
            "uniform_regret_per_step",
            report.uniform_regret_per_step,
            constraints.min_uniform_regret_per_step,
            constraints.max_uniform_regret_per_step,
        ),
        (
            "mean_q_gap",
            report.mean_q_gap,
            constraints.min_mean_q_gap,
            constraints.max_mean_q_gap,
        ),
        (
            "mean_depth_entropy",
            report.mean_depth_entropy,
            constraints.min_mean_depth_entropy,
            None,
        ),
    )
    violations: list[str] = []
    for name, value, lower, upper in checks:
        if lower is not None and value < lower:
            violations.append(f"{name}={value:.6g} < {lower:.6g}")
        if upper is not None and value > upper:
            violations.append(f"{name}={value:.6g} > {upper:.6g}")
    return tuple(violations)


def synthesize_mdp(
    generator: GeneratorConfig,
    difficulty: DifficultyConfig,
    *,
    seed: int,
    mdp_id: str,
) -> tuple[PuzzleMDP, DifficultyReport, int]:
    """Rejection-sample a task whose *measured* difficulty meets the spec."""

    seed_sequence = np.random.SeedSequence(seed)
    attempt_sequences = seed_sequence.spawn(difficulty.max_generation_attempts)
    last_violations: tuple[str, ...] = ()
    for attempt, attempt_sequence in enumerate(attempt_sequences, start=1):
        candidate_seed = int(attempt_sequence.generate_state(1, dtype=np.uint64)[0])
        candidate = generate_layered_mdp(generator, seed=candidate_seed, mdp_id=mdp_id)
        report = measure_difficulty(candidate)
        last_violations = difficulty_violations(report, difficulty)
        if not last_violations:
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "synthesis_seed": seed,
                    "synthesis_attempt": attempt,
                    "difficulty": report.to_dict(),
                    "difficulty_constraints": asdict(difficulty),
                }
            )
            return replace(candidate, metadata=metadata), report, attempt
    raise ConfigurationError(
        f"failed to synthesize {mdp_id!r} after {difficulty.max_generation_attempts} "
        f"attempts; last violations: {list(last_violations)}"
    )
