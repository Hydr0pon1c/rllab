"""Deterministic, balanced layered-DAG PuzzleMDP generation."""

from __future__ import annotations

from dataclasses import asdict
from math import ceil

import numpy as np

from rllab.config import GeneratorConfig
from rllab.errors import ConfigurationError
from rllab.mdp.schema import PuzzleMDP, State, Transition, validate_mdp


def layer_widths(config: GeneratorConfig) -> tuple[int, ...]:
    """Return reachable widths for depths 0..horizon."""

    widths = [1]
    for _depth in range(1, config.horizon):
        widths.append(min(config.width, widths[-1] * config.num_actions))
    terminal_width = (
        config.terminal_width
        if config.terminal_width is not None
        else min(config.width, widths[-1] * config.num_actions)
    )
    if terminal_width > widths[-1] * config.num_actions:
        raise ConfigurationError(
            "terminal_width exceeds the incoming action capacity at the final layer"
        )
    widths.append(terminal_width)
    if any(width < config.num_actions for width in widths[1:]):
        raise ConfigurationError(
            "every non-start layer must contain at least num_actions states"
        )
    return tuple(widths)


def _balanced_targets(
    num_sources: int,
    num_targets: int,
    num_actions: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Assign distinct targets per state, full coverage, and balanced in-degree."""

    capacity = num_sources * num_actions
    if num_targets > capacity:
        raise ConfigurationError("next layer is wider than the transition capacity")
    if num_targets < num_actions:
        raise ConfigurationError(
            "next layer must have at least num_actions distinct states"
        )

    slots = [
        (source, slot) for source in range(num_sources) for slot in range(num_actions)
    ]
    rng.shuffle(slots)
    targets = [[-1] * num_actions for _ in range(num_sources)]
    used: list[set[int]] = [set() for _ in range(num_sources)]
    indegree = np.zeros(num_targets, dtype=np.int64)

    # First place each target once. Because targets are unique in this phase,
    # no source can receive the same target twice.
    destination_order = rng.permutation(num_targets).tolist()
    for (source, slot), target in zip(
        slots[:num_targets], destination_order, strict=True
    ):
        targets[source][slot] = target
        used[source].add(target)
        indegree[target] += 1

    # Fill remaining slots from the least-used valid destinations. This avoids
    # target-frequency leakage while retaining random topology.
    for source, slot in slots[num_targets:]:
        candidates = np.array(
            [target for target in range(num_targets) if target not in used[source]],
            dtype=np.int64,
        )
        candidate_degrees = indegree[candidates]
        least_used = candidates[candidate_degrees == candidate_degrees.min()]
        target = int(rng.choice(least_used))
        targets[source][slot] = target
        used[source].add(target)
        indegree[target] += 1

    return targets


def _sample_costs_with_gap(
    downstream_costs: np.ndarray,
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample displayed integer costs until the best total cost has a set gap."""

    for _ in range(10_000):
        costs = rng.integers(
            config.min_cost,
            config.max_cost + 1,
            size=config.num_actions,
            dtype=np.int64,
        )
        totals = costs.astype(np.float64) + downstream_costs
        ordered = np.sort(totals)
        if ordered[1] - ordered[0] >= config.min_optimality_gap:
            return costs, totals
    raise ConfigurationError(
        "could not sample a unique-enough best action; increase the cost range or reduce "
        "generator.min_optimality_gap"
    )


def generate_layered_mdp(
    config: GeneratorConfig,
    *,
    seed: int,
    mdp_id: str | None = None,
) -> PuzzleMDP:
    """Generate one exact, static PuzzleMDP.

    Action positions are assigned after solving the unlabeled graph. Sorting
    outgoing edges by exact optimal cost and rotating ranks across states makes
    every action slot receive every rank equally often (up to one state).
    """

    config.validate()
    widths = layer_widths(config)
    rng = np.random.default_rng(seed)

    layers: list[list[str]] = []
    states: list[State] = []
    state_counter = 0
    for depth, width in enumerate(widths):
        layer: list[str] = []
        for _ in range(width):
            state_id = f"s{state_counter:05d}"
            state_counter += 1
            layer.append(state_id)
            is_terminal = depth == config.horizon
            terminal_cost = (
                int(
                    rng.integers(config.terminal_cost_min, config.terminal_cost_max + 1)
                )
                if is_terminal
                else 0
            )
            states.append(
                State(
                    state_id=state_id,
                    depth=depth,
                    terminal=is_terminal,
                    terminal_cost=terminal_cost,
                )
            )
        layers.append(layer)

    target_ids: dict[str, list[str]] = {}
    for depth in range(config.horizon):
        assignments = _balanced_targets(
            len(layers[depth]),
            len(layers[depth + 1]),
            config.num_actions,
            rng,
        )
        for source_index, source in enumerate(layers[depth]):
            target_ids[source] = [
                layers[depth + 1][target_index]
                for target_index in assignments[source_index]
            ]

    state_by_id = {state.state_id: state for state in states}
    optimal_cost: dict[str, float] = {
        state_id: float(state_by_id[state_id].terminal_cost) for state_id in layers[-1]
    }
    pending_edges: dict[str, list[tuple[str, int, float]]] = {}
    for depth in reversed(range(config.horizon)):
        for source in layers[depth]:
            targets = target_ids[source]
            downstream = np.asarray(
                [optimal_cost[target] for target in targets], dtype=np.float64
            )
            costs, totals = _sample_costs_with_gap(downstream, config, rng)
            pending_edges[source] = [
                (target, int(cost), float(total))
                for target, cost, total in zip(targets, costs, totals, strict=True)
            ]
            optimal_cost[source] = float(totals.min())

    nonterminal_states = [state_id for layer in layers[:-1] for state_id in layer]
    randomized_states = np.asarray(nonterminal_states, dtype=object)
    rng.shuffle(randomized_states)
    best_slots = np.tile(
        np.arange(config.num_actions, dtype=np.int64),
        ceil(len(nonterminal_states) / config.num_actions),
    )[: len(nonterminal_states)]
    rng.shuffle(best_slots)
    slot_for_state = {
        str(state_id): int(slot)
        for state_id, slot in zip(
            randomized_states.tolist(), best_slots.tolist(), strict=True
        )
    }

    transitions: list[Transition] = []
    optimal_action_counts = [0] * config.num_actions
    for source in nonterminal_states:
        ranked = sorted(pending_edges[source], key=lambda edge: (edge[2], edge[0]))
        rotation = slot_for_state[source]
        for rank, (target, cost, _total) in enumerate(ranked):
            action = (rank + rotation) % config.num_actions
            transitions.append(
                Transition(source=source, action=action, target=target, cost=cost)
            )
            if rank == 0:
                optimal_action_counts[action] += 1

    resolved_id = mdp_id or f"mdp-s{seed}"
    mdp = PuzzleMDP(
        mdp_id=resolved_id,
        seed=seed,
        start_state=layers[0][0],
        horizon=config.horizon,
        num_actions=config.num_actions,
        states=tuple(states),
        transitions=tuple(transitions),
        metadata={
            "generator": "layered-balanced/v1",
            "generator_config": asdict(config),
            "layer_widths": list(widths),
            "optimal_action_counts": optimal_action_counts,
        },
    )
    validate_mdp(mdp)
    if max(optimal_action_counts) - min(optimal_action_counts) > 1:
        raise AssertionError("internal error: optimal actions are not balanced")
    return mdp
