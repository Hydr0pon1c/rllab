"""Directional cross-state policy-update kernel diagnostics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from rllab.data.exposure import classify_state_exposure
from rllab.experiment.corpus import ExperimentCorpus
from rllab.experiment.evaluation import PolicySnapshot
from rllab.mdp.solve import (
    evaluate_policy,
    mdp_arrays,
    occupancy_from,
    performance_difference,
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return None
    return float(np.dot(left, right) / denominator)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def directional_kernel_metrics(
    corpus: ExperimentCorpus,
    before: PolicySnapshot,
    fractional: PolicySnapshot,
    after: PolicySnapshot,
    *,
    probe_fraction: float,
    policy_loss_states: Mapping[str, Iterable[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare a fractional-step secant prediction with the full SGD update.

    If ``theta_e = theta - f * eta * grad``, then
    ``(pi(theta_e)-pi(theta))/f`` estimates the full-step first-order kernel
    response. The full update follows the same frozen gradient direction.
    """

    if not 0.0 < probe_fraction < 1.0:
        raise ValueError("probe_fraction must be in (0, 1)")
    rows: list[dict[str, Any]] = []
    flat_predicted: list[float] = []
    flat_actual: list[float] = []

    for bundle in corpus.bundles:
        mdp = bundle.mdp
        mdp_id = mdp.mdp_id
        arrays = mdp_arrays(mdp)
        base = before.policy(mdp_id)
        probe = fractional.policy(mdp_id)
        updated = after.policy(mdp_id)
        predicted_delta = (probe - base) / probe_fraction
        actual_delta = updated - base
        base_eval = evaluate_policy(mdp, base)
        exact_difference = performance_difference(mdp, base, updated)
        exposure = {
            item.state_id: item
            for item in classify_state_exposure(mdp, policy_loss_states.get(mdp_id, ()))
        }

        per_state_predicted_advantage = np.nansum(
            predicted_delta * base_eval.advantage, axis=1
        )
        for state_index, state_id in enumerate(arrays.state_ids):
            state_exposure = exposure[state_id]
            if arrays.terminal[state_index]:
                predicted = np.zeros(mdp.num_actions, dtype=np.float64)
                actual = np.zeros(mdp.num_actions, dtype=np.float64)
            else:
                predicted = predicted_delta[state_index]
                actual = actual_delta[state_index]
                flat_predicted.extend(predicted.tolist())
                flat_actual.extend(actual.tolist())
            predicted_value_delta = float(
                np.dot(
                    occupancy_from(mdp, base, state_id),
                    per_state_predicted_advantage,
                )
            )
            rows.append(
                {
                    "schema_version": "directional-kernel-state/v1",
                    "mdp_id": mdp_id,
                    "state_id": state_id,
                    "depth": int(arrays.depths[state_index]),
                    "terminal": bool(arrays.terminal[state_index]),
                    "exposure_class": state_exposure.exposure_class.value,
                    "predicted_policy_delta": [float(value) for value in predicted],
                    "actual_policy_delta": [float(value) for value in actual],
                    "policy_delta_rmse": float(
                        np.sqrt(np.mean((predicted - actual) ** 2))
                    ),
                    "policy_delta_cosine": _cosine(predicted, actual),
                    "predicted_delta_value_first_order": predicted_value_delta,
                    "actual_delta_value": float(
                        exact_difference.exact_delta_j[state_index]
                    ),
                    "value_prediction_error": float(
                        predicted_value_delta
                        - exact_difference.exact_delta_j[state_index]
                    ),
                }
            )

    predicted_array = np.asarray(flat_predicted, dtype=np.float64)
    actual_array = np.asarray(flat_actual, dtype=np.float64)
    aggregate: dict[str, Any] = {
        "probe_fraction": probe_fraction,
        "policy_delta_rmse": float(
            np.sqrt(np.mean((predicted_array - actual_array) ** 2))
        ),
        "policy_delta_mae": float(np.mean(np.abs(predicted_array - actual_array))),
        "policy_delta_cosine": _cosine(predicted_array, actual_array),
        "policy_delta_correlation": _correlation(predicted_array, actual_array),
    }
    exposure_summary: dict[str, Any] = {}
    for name in ("policy_loss_current", "destination_context_only", "token_absent"):
        selected = [
            row for row in rows if row["exposure_class"] == name and not row["terminal"]
        ]
        exposure_summary[name] = {
            "states": len(selected),
            "mean_policy_delta_rmse": (
                float(np.mean([row["policy_delta_rmse"] for row in selected]))
                if selected
                else None
            ),
            "mean_actual_policy_l2": (
                float(
                    np.mean(
                        [np.linalg.norm(row["actual_policy_delta"]) for row in selected]
                    )
                )
                if selected
                else None
            ),
            "mean_actual_delta_value": (
                float(np.mean([row["actual_delta_value"] for row in selected]))
                if selected
                else None
            ),
            "mean_abs_value_prediction_error": (
                float(np.mean([abs(row["value_prediction_error"]) for row in selected]))
                if selected
                else None
            ),
        }
    aggregate["exposure"] = exposure_summary
    return rows, aggregate
