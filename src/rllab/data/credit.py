"""Outcome and exact-process credit views over the same trajectory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rllab.data.trajectory import Trajectory
from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP
from rllab.mdp.solve import evaluate_policy, mdp_arrays, validate_policy

CREDIT_SCHEMA_VERSION = "trajectory-credit/v1"


@dataclass(frozen=True)
class StepCredit:
    step: int
    state_id: str
    action: int
    environment_reward: float
    realised_return_to_go: float
    outcome_return: float
    exact_value: float
    exact_q: float
    exact_advantage: float


@dataclass(frozen=True)
class TrajectoryCredit:
    mdp_id: str
    total_return: float
    steps: tuple[StepCredit, ...]
    schema_version: str = CREDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mdp_id": self.mdp_id,
            "total_return": self.total_return,
            "steps": [asdict(step) for step in self.steps],
        }


def trajectory_credit(
    mdp: PuzzleMDP,
    trajectory: Trajectory,
    reference_policy: NDArray[np.floating],
) -> TrajectoryCredit:
    """Materialise ORM outcome and unbiased exact process targets side by side."""

    if trajectory.mdp_id != mdp.mdp_id:
        raise ValidationError("trajectory and MDP IDs do not match")
    arrays = mdp_arrays(mdp)
    policy = validate_policy(mdp, reference_policy)
    evaluation = evaluate_policy(mdp, policy)
    rewards = [-float(step.immediate_cost) for step in trajectory.steps]
    if rewards:
        rewards[-1] -= trajectory.terminal_cost
    realised_returns = np.cumsum(np.asarray(rewards[::-1], dtype=np.float64))[::-1]
    total_return = -float(trajectory.total_cost)
    credits: list[StepCredit] = []
    for index, step in enumerate(trajectory.steps):
        state_index = arrays.state_index[step.state_id]
        credits.append(
            StepCredit(
                step=step.step,
                state_id=step.state_id,
                action=step.action,
                environment_reward=rewards[index],
                realised_return_to_go=float(realised_returns[index]),
                outcome_return=total_return,
                exact_value=float(evaluation.value[state_index]),
                exact_q=float(evaluation.q_value[state_index, step.action]),
                exact_advantage=float(evaluation.advantage[state_index, step.action]),
            )
        )
    return TrajectoryCredit(
        mdp_id=mdp.mdp_id,
        total_return=total_return,
        steps=tuple(credits),
    )
