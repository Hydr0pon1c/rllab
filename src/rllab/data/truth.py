"""Normalized exact-ground-truth records for versioned datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from rllab.mdp.schema import PuzzleMDP
from rllab.mdp.solve import evaluate_policy, mdp_arrays, optimal_policy, uniform_policy

TRUTH_SCHEMA_VERSION = "exact-truth/v1"


def exact_truth_records(mdp: PuzzleMDP) -> tuple[dict[str, Any], ...]:
    """Materialize per-state V/Q/A, occupancy, regret, and exact PG targets."""

    arrays = mdp_arrays(mdp)
    uniform = uniform_policy(mdp)
    optimal = optimal_policy(mdp)
    uniform_eval = evaluate_policy(mdp, uniform)
    optimal_eval = evaluate_policy(mdp, optimal)
    outgoing = mdp.outgoing()
    records: list[dict[str, Any]] = []

    for state_index, state_id in enumerate(arrays.state_ids):
        terminal = bool(arrays.terminal[state_index])
        actions: list[dict[str, Any]] = []
        if not terminal:
            optimal_q = optimal_eval.q_value[state_index]
            best_q = float(np.max(optimal_q))
            by_action = {
                transition.action: transition for transition in outgoing[state_id]
            }
            for action in range(mdp.num_actions):
                transition = by_action[action]
                actions.append(
                    {
                        "action": action,
                        "target": transition.target,
                        "immediate_cost": transition.cost,
                        "uniform_probability": float(uniform[state_index, action]),
                        "uniform_q": float(uniform_eval.q_value[state_index, action]),
                        "uniform_advantage": float(
                            uniform_eval.advantage[state_index, action]
                        ),
                        "root_uniform_logit_gradient": float(
                            uniform_eval.logit_gradient[state_index, action]
                        ),
                        "optimal_q": float(optimal_q[action]),
                        "optimal_cost_to_go": float(-optimal_q[action]),
                        "optimal_regret": float(best_q - optimal_q[action]),
                        "is_optimal": bool(
                            np.isclose(optimal_q[action], best_q, atol=1e-12, rtol=0.0)
                        ),
                    }
                )
        records.append(
            {
                "schema_version": TRUTH_SCHEMA_VERSION,
                "mdp_id": mdp.mdp_id,
                "state_id": state_id,
                "depth": int(arrays.depths[state_index]),
                "terminal": terminal,
                "terminal_cost": float(arrays.terminal_costs[state_index]),
                "uniform_value": float(uniform_eval.value[state_index]),
                "uniform_cost_to_go": float(-uniform_eval.value[state_index]),
                "optimal_value": float(optimal_eval.value[state_index]),
                "optimal_cost_to_go": float(-optimal_eval.value[state_index]),
                "root_uniform_occupancy": float(uniform_eval.occupancy[state_index]),
                "actions": actions,
            }
        )
    return tuple(records)
