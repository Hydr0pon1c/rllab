"""Deterministic-environment trajectory records and policy sampling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rllab.mdp.schema import PuzzleMDP
from rllab.mdp.solve import mdp_arrays, validate_policy

TRAJECTORY_SCHEMA_VERSION = "trajectory/v1"


@dataclass(frozen=True)
class TrajectoryStep:
    step: int
    state_id: str
    action: int
    target_state_id: str
    immediate_cost: int


@dataclass(frozen=True)
class Trajectory:
    mdp_id: str
    start_state: str
    terminal_state: str
    steps: tuple[TrajectoryStep, ...]
    transition_cost: int
    terminal_cost: int
    total_cost: int
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mdp_id": self.mdp_id,
            "start_state": self.start_state,
            "terminal_state": self.terminal_state,
            "steps": [asdict(step) for step in self.steps],
            "transition_cost": self.transition_cost,
            "terminal_cost": self.terminal_cost,
            "total_cost": self.total_cost,
        }


def sample_trajectory(
    mdp: PuzzleMDP,
    policy: NDArray[np.floating],
    *,
    seed: int,
    start_state: str | None = None,
) -> Trajectory:
    arrays = mdp_arrays(mdp)
    checked = validate_policy(mdp, policy)
    rng = np.random.default_rng(seed)
    state_by_id = {state.state_id: state for state in mdp.states}
    outgoing = mdp.outgoing()
    current = start_state or mdp.start_state
    initial = current
    steps: list[TrajectoryStep] = []
    transition_cost = 0

    while not state_by_id[current].terminal:
        index = arrays.state_index[current]
        action = int(rng.choice(mdp.num_actions, p=checked[index]))
        transition = outgoing[current][action]
        steps.append(
            TrajectoryStep(
                step=len(steps),
                state_id=current,
                action=action,
                target_state_id=transition.target,
                immediate_cost=transition.cost,
            )
        )
        transition_cost += transition.cost
        current = transition.target

    terminal_cost = state_by_id[current].terminal_cost
    return Trajectory(
        mdp_id=mdp.mdp_id,
        start_state=initial,
        terminal_state=current,
        steps=tuple(steps),
        transition_cost=transition_cost,
        terminal_cost=terminal_cost,
        total_cost=transition_cost + terminal_cost,
    )
