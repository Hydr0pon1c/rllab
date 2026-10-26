"""CPU finite-response, attribution and exact-objective analysis utilities."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rllab.data.exposure import classify_state_exposure
from rllab.experiment.evaluation import PolicySnapshot
from rllab.mdp.solve import evaluate_policy, mdp_arrays, performance_difference


def _finite_policy_terms(
    corpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
) -> dict[str, Any]:
    """Return exact finite policy and occupancy terms without parameter dots."""

    expected = {bundle.mdp.mdp_id for bundle in corpus.bundles}
    if set(before.policies) != expected or set(after.policies) != expected:
        raise ValueError("policy snapshots do not cover the experiment MDPs")

    per_mdp: dict[str, dict[str, float]] = {}
    for bundle in corpus.bundles:
        mdp = bundle.mdp
        mdp_id = mdp.mdp_id
        base_policy = before.policy(mdp_id)
        updated_policy = after.policy(mdp_id)
        arrays = mdp_arrays(mdp)
        base = evaluate_policy(mdp, base_policy)
        updated = evaluate_policy(mdp, updated_policy)
        policy_delta = updated_policy - base_policy
        per_state_policy_gain = np.nansum(policy_delta * base.advantage, axis=1)
        fixed_occupancy_gain = float(np.dot(base.occupancy, per_state_policy_gain))
        actual_gain = float(updated.objective - base.objective)
        identity = performance_difference(
            mdp, base_policy, updated_policy, starts=(mdp.start_state,)
        )
        nonterminal = ~arrays.terminal
        per_mdp[mdp_id] = {
            "actual_gain": actual_gain,
            "fixed_occupancy_policy_gain": fixed_occupancy_gain,
            "occupancy_redistribution": actual_gain - fixed_occupancy_gain,
            "performance_identity_residual": float(identity.residual[0]),
            "policy_response_l2": float(
                np.linalg.norm(policy_delta[nonterminal].reshape(-1))
            ),
            "min_action_probability": float(updated_policy[nonterminal].min()),
            "max_action_probability": float(updated_policy[nonterminal].max()),
        }

    aggregate = {
        field: float(np.mean([row[field] for row in per_mdp.values()]))
        for field in (
            "actual_gain",
            "fixed_occupancy_policy_gain",
            "occupancy_redistribution",
            "policy_response_l2",
        )
    }
    aggregate.update(
        {
            "max_abs_performance_identity_residual": max(
                abs(row["performance_identity_residual"]) for row in per_mdp.values()
            ),
            "min_action_probability": min(
                row["min_action_probability"] for row in per_mdp.values()
            ),
            "max_action_probability": max(
                row["max_action_probability"] for row in per_mdp.values()
            ),
        }
    )
    return {"aggregate": aggregate, "per_mdp": per_mdp}


TARGET_EXPOSURE_CLASSES = (
    "policy_loss_current",
    "destination_context_only",
    "token_absent",
)


def finite_set_shapley(
    values: Mapping[int, float], player_names: Sequence[str]
) -> dict[str, Any]:
    """Exactly attribute a complete finite set function.

    Integer keys are subset bitmasks. The returned Shapley values sum to the
    full-minus-empty effect, while the Mobius dividends expose whether that
    effect is additive or depends on higher-order component interactions.
    """

    names = tuple(str(name) for name in player_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("player_names must be non-empty and unique")
    count = len(names)
    expected = set(range(1 << count))
    if set(values) != expected:
        raise ValueError("finite set function must contain every subset mask")
    game = {mask: float(values[mask]) for mask in expected}
    if not all(math.isfinite(value) for value in game.values()):
        raise ValueError("finite set function contains a non-finite value")

    factorial = [math.factorial(index) for index in range(count + 1)]
    denominator = factorial[count]
    shapley: dict[str, float] = {}
    for player, name in enumerate(names):
        bit = 1 << player
        value = 0.0
        for mask in expected:
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = factorial[size] * factorial[count - size - 1] / denominator
            value += weight * (game[mask | bit] - game[mask])
        shapley[name] = value

    dividends: dict[int, float] = {}
    for mask in sorted(expected, key=lambda item: (item.bit_count(), item)):
        proper = (mask - 1) & mask
        contained = [0] if mask else []
        while proper:
            contained.append(proper)
            proper = (proper - 1) & mask
        dividends[mask] = game[mask] - math.fsum(dividends[item] for item in contained)
    by_order = {
        str(order): math.fsum(
            value for mask, value in dividends.items() if mask.bit_count() == order
        )
        for order in range(count + 1)
    }
    total = game[(1 << count) - 1] - game[0]
    player_rows = {
        name: {
            "value": value,
            "signed_share": value / total if total != 0.0 else None,
        }
        for name, value in shapley.items()
    }
    terms = [
        {
            "mask": mask,
            "players": [
                name for index, name in enumerate(names) if mask & (1 << index)
            ],
            "order": mask.bit_count(),
            "value": dividends[mask],
        }
        for mask in sorted(expected)
    ]
    return {
        "empty": game[0],
        "full": game[(1 << count) - 1],
        "total_effect": total,
        "players": player_rows,
        "efficiency_residual": total - math.fsum(shapley.values()),
        "mobius_by_order": by_order,
        "mobius_terms": terms,
    }


def finite_policy_map_components(
    singleton_cells: Sequence[Mapping[str, float]],
    *,
    additive_logit_fixed_gain: float,
    full_actual_gain: float,
    full_fixed_gain: float,
) -> dict[str, float]:
    """Exactly split a full finite gain around singleton logit responses.

    Each singleton row supplies ``actual_gain`` and
    ``fixed_occupancy_policy_gain``. The synthetic additive-logit policy is a
    valid softmax policy evaluated with the same fixed-base-occupancy readout.
    """

    if not singleton_cells:
        raise ValueError("at least one singleton cell is required")
    singleton_actual = math.fsum(float(row["actual_gain"]) for row in singleton_cells)
    singleton_fixed = math.fsum(
        float(row["fixed_occupancy_policy_gain"]) for row in singleton_cells
    )
    additive_fixed = float(additive_logit_fixed_gain)
    full_actual = float(full_actual_gain)
    full_fixed = float(full_fixed_gain)
    components = {
        "singleton": singleton_actual,
        "softmax": additive_fixed - singleton_fixed,
        "logit_map": full_fixed - additive_fixed,
        "occupancy": (full_actual - full_fixed) - (singleton_actual - singleton_fixed),
    }
    components["total"] = math.fsum(components.values())
    components["identity_residual"] = full_actual - components["total"]
    return components


def finite_target_exposure_terms(
    corpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
    *,
    policy_loss_states: Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    """Decompose one exact finite objective gain by target-state exposure.

    For each target state this uses the exact performance-difference summand
    ``d^{pi'}(s) * sum_a (pi' - pi)(a|s) * A^pi(s,a)``. The fixed-occupancy
    counterpart isolates the finite parameter-to-policy response from the
    subsequent occupancy redistribution. Aggregates use the same equal-MDP
    mean as the training objective; policy-response norms cover all decision
    states without averaging.
    """

    expected = {bundle.mdp.mdp_id for bundle in corpus.bundles}
    if set(before.policies) != expected or set(after.policies) != expected:
        raise ValueError("policy snapshots do not cover the experiment MDPs")

    per_mdp: dict[str, dict[str, dict[str, float | int]]] = {}
    max_identity_residual = 0.0
    for bundle in corpus.bundles:
        mdp = bundle.mdp
        mdp_id = mdp.mdp_id
        arrays = mdp_arrays(mdp)
        base_policy = before.policy(mdp_id)
        updated_policy = after.policy(mdp_id)
        base = evaluate_policy(mdp, base_policy)
        updated = evaluate_policy(mdp, updated_policy)
        policy_delta = updated_policy - base_policy
        advantage_response = np.nansum(policy_delta * base.advantage, axis=1)
        fixed_contribution = base.occupancy * advantage_response
        actual_contribution = updated.occupancy * advantage_response
        exposure = {
            row.state_id: row.exposure_class.value
            for row in classify_state_exposure(mdp, policy_loss_states.get(mdp_id, ()))
        }
        groups: dict[str, dict[str, float | int]] = {}
        for name in TARGET_EXPOSURE_CLASSES:
            selected = np.asarray(
                [
                    not arrays.terminal[index] and exposure[state_id] == name
                    for index, state_id in enumerate(arrays.state_ids)
                ],
                dtype=bool,
            )
            fixed_gain = float(np.sum(fixed_contribution[selected]))
            actual_gain = float(np.sum(actual_contribution[selected]))
            groups[name] = {
                "states": int(np.count_nonzero(selected)),
                "actual_gain": actual_gain,
                "fixed_occupancy_policy_gain": fixed_gain,
                "occupancy_redistribution": actual_gain - fixed_gain,
                "policy_response_l2_square": float(
                    np.square(policy_delta[selected]).sum(dtype=np.float64)
                ),
            }
        exact_gain = float(updated.objective - base.objective)
        grouped_gain = math.fsum(
            float(groups[name]["actual_gain"]) for name in TARGET_EXPOSURE_CLASSES
        )
        residual = exact_gain - grouped_gain
        max_identity_residual = max(max_identity_residual, abs(residual))
        per_mdp[mdp_id] = {
            "groups": groups,
            "exact_gain": exact_gain,
            "grouped_gain": grouped_gain,
            "identity_residual": residual,
        }

    aggregate_groups: dict[str, dict[str, float | int]] = {}
    mdp_count = len(corpus.bundles)
    for name in TARGET_EXPOSURE_CLASSES:
        rows = [per_mdp[bundle.mdp.mdp_id]["groups"][name] for bundle in corpus.bundles]
        aggregate_groups[name] = {
            "states": sum(int(row["states"]) for row in rows),
            "actual_gain": math.fsum(float(row["actual_gain"]) for row in rows)
            / mdp_count,
            "fixed_occupancy_policy_gain": math.fsum(
                float(row["fixed_occupancy_policy_gain"]) for row in rows
            )
            / mdp_count,
            "occupancy_redistribution": math.fsum(
                float(row["occupancy_redistribution"]) for row in rows
            )
            / mdp_count,
            "policy_response_l2": math.sqrt(
                math.fsum(float(row["policy_response_l2_square"]) for row in rows)
            ),
        }

    exact_gain = (
        math.fsum(float(row["exact_gain"]) for row in per_mdp.values()) / mdp_count
    )
    grouped_gain = math.fsum(
        float(aggregate_groups[name]["actual_gain"]) for name in TARGET_EXPOSURE_CLASSES
    )
    return {
        "groups": aggregate_groups,
        "aggregate": {
            "actual_gain": exact_gain,
            "grouped_gain": grouped_gain,
            "identity_residual": exact_gain - grouped_gain,
            "max_abs_per_mdp_identity_residual": max_identity_residual,
            "policy_response_l2": math.sqrt(
                math.fsum(
                    float(aggregate_groups[name]["policy_response_l2"]) ** 2
                    for name in TARGET_EXPOSURE_CLASSES
                )
            ),
        },
        "per_mdp": per_mdp,
    }


def aggregate_policy_update_decomposition(
    corpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
    *,
    exact_opportunity: float,
    nominal_projection: float,
    realized_projection: float,
) -> dict[str, Any]:
    """Split aggregate task gain while retaining per-MDP finite policy terms."""

    if not all(
        math.isfinite(value)
        for value in (exact_opportunity, nominal_projection, realized_projection)
    ):
        raise ValueError("aggregate parameter projections must be finite")
    finite = _finite_policy_terms(corpus, before, after)
    aggregate = dict(finite["aggregate"])
    projected_credit = nominal_projection - exact_opportunity
    materialization = realized_projection - nominal_projection
    policy_map = aggregate["fixed_occupancy_policy_gain"] - realized_projection
    occupancy = aggregate["occupancy_redistribution"]
    residual = aggregate["actual_gain"] - (
        exact_opportunity + projected_credit + materialization + policy_map + occupancy
    )
    aggregate.update(
        {
            "exact_direction_opportunity": exact_opportunity,
            "nominal_projected_credit": projected_credit,
            "nominal_parameter_projection": nominal_projection,
            "materialization": materialization,
            "realized_parameter_projection": realized_projection,
            "policy_map_curvature": policy_map,
            "additive_residual": residual,
            "max_abs_additive_residual": abs(residual),
        }
    )
    return {"aggregate": aggregate, "per_mdp": finite["per_mdp"]}


def _flat_response_metrics(
    reference_values: list[np.ndarray], candidate_values: list[np.ndarray]
) -> dict[str, float | int | None]:
    if not reference_values or len(reference_values) != len(candidate_values):
        raise ValueError("finite-response groups must contain paired states")
    reference = np.concatenate(reference_values).astype(np.float64, copy=False)
    candidate = np.concatenate(candidate_values).astype(np.float64, copy=False)
    error = candidate - reference
    reference_l2 = float(np.linalg.norm(reference))
    candidate_l2 = float(np.linalg.norm(candidate))
    denominator = reference_l2 * candidate_l2
    cosine = float(np.dot(reference, candidate) / denominator) if denominator else None
    correlation = (
        float(np.corrcoef(reference, candidate)[0, 1])
        if reference.size > 1
        and float(reference.std()) > 0.0
        and float(candidate.std()) > 0.0
        else None
    )
    return {
        "states": len(reference_values),
        "action_elements": int(reference.size),
        "reference_l2": reference_l2,
        "candidate_l2": candidate_l2,
        "candidate_to_reference_l2_ratio": (
            candidate_l2 / reference_l2 if reference_l2 > 0.0 else None
        ),
        "error_l2": float(np.linalg.norm(error)),
        "relative_l2_error": (
            float(np.linalg.norm(error)) / reference_l2 if reference_l2 > 0.0 else None
        ),
        "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "correlation": correlation,
    }


def finite_policy_response_comparison(
    corpus,
    before: PolicySnapshot,
    reference_after: PolicySnapshot,
    candidate_after: PolicySnapshot,
    *,
    policy_loss_states: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a sampled finite policy update with an exact-direction update."""

    groups: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray]]]] = {
        "mdp": {},
        "depth": {},
        "exposure": {},
    }
    all_reference: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []

    def append_group(kind: str, key: str, reference, candidate) -> None:
        selected = groups[kind].setdefault(key, ([], []))
        selected[0].append(reference)
        selected[1].append(candidate)

    for bundle in corpus.bundles:
        mdp = bundle.mdp
        mdp_id = mdp.mdp_id
        arrays = mdp_arrays(mdp)
        base = before.policy(mdp_id)
        reference_delta = reference_after.policy(mdp_id) - base
        candidate_delta = candidate_after.policy(mdp_id) - base
        exposure = {
            item.state_id: item.exposure_class.value
            for item in classify_state_exposure(mdp, policy_loss_states.get(mdp_id, ()))
        }
        for state_index, state_id in enumerate(arrays.state_ids):
            if arrays.terminal[state_index]:
                continue
            reference = reference_delta[state_index].reshape(-1)
            candidate = candidate_delta[state_index].reshape(-1)
            all_reference.append(reference)
            all_candidate.append(candidate)
            append_group("mdp", mdp_id, reference, candidate)
            append_group(
                "depth", str(int(arrays.depths[state_index])), reference, candidate
            )
            append_group("exposure", exposure[state_id], reference, candidate)

    return {
        "aggregate": _flat_response_metrics(all_reference, all_candidate),
        **{
            kind: {
                key: _flat_response_metrics(reference, candidate)
                for key, (reference, candidate) in selected.items()
            }
            for kind, selected in groups.items()
        },
    }


def policy_update_decomposition(
    corpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
    *,
    exact_opportunity_by_mdp: Mapping[str, float],
    nominal_projection_by_mdp: Mapping[str, float],
    realized_projection_by_mdp: Mapping[str, float],
) -> dict[str, Any]:
    """Split each exact objective change into five signed additive terms."""

    expected = {bundle.mdp.mdp_id for bundle in corpus.bundles}
    for name, values in (
        ("exact opportunity", exact_opportunity_by_mdp),
        ("nominal projection", nominal_projection_by_mdp),
        ("realized projection", realized_projection_by_mdp),
    ):
        if set(values) != expected:
            raise ValueError(f"{name} does not cover the experiment MDPs")

    finite = _finite_policy_terms(corpus, before, after)
    per_mdp: dict[str, dict[str, float]] = {}
    for bundle in corpus.bundles:
        mdp_id = bundle.mdp.mdp_id
        finite_row = finite["per_mdp"][mdp_id]
        actual_gain = finite_row["actual_gain"]
        fixed_occupancy_gain = finite_row["fixed_occupancy_policy_gain"]
        exact_opportunity = float(exact_opportunity_by_mdp[mdp_id])
        nominal_projection = float(nominal_projection_by_mdp[mdp_id])
        realized_projection = float(realized_projection_by_mdp[mdp_id])
        projected_credit = nominal_projection - exact_opportunity
        materialization = realized_projection - nominal_projection
        policy_curvature = fixed_occupancy_gain - realized_projection
        occupancy = actual_gain - fixed_occupancy_gain
        additive_residual = actual_gain - (
            exact_opportunity
            + projected_credit
            + materialization
            + policy_curvature
            + occupancy
        )
        per_mdp[mdp_id] = {
            **finite_row,
            "exact_direction_opportunity": exact_opportunity,
            "nominal_projected_credit": projected_credit,
            "nominal_parameter_projection": nominal_projection,
            "materialization": materialization,
            "realized_parameter_projection": realized_projection,
            "policy_map_curvature": policy_curvature,
            "occupancy_redistribution": occupancy,
            "additive_residual": additive_residual,
        }

    mean_fields = (
        "actual_gain",
        "exact_direction_opportunity",
        "nominal_projected_credit",
        "nominal_parameter_projection",
        "materialization",
        "realized_parameter_projection",
        "policy_map_curvature",
        "fixed_occupancy_policy_gain",
        "occupancy_redistribution",
        "additive_residual",
        "policy_response_l2",
    )
    aggregate = {
        field: float(np.mean([row[field] for row in per_mdp.values()]))
        for field in mean_fields
    }
    aggregate.update(
        {
            "max_abs_additive_residual": max(
                abs(row["additive_residual"]) for row in per_mdp.values()
            ),
            "max_abs_performance_identity_residual": max(
                abs(row["performance_identity_residual"]) for row in per_mdp.values()
            ),
            "min_action_probability": min(
                row["min_action_probability"] for row in per_mdp.values()
            ),
            "max_action_probability": max(
                row["max_action_probability"] for row in per_mdp.values()
            ),
        }
    )
    return {"aggregate": aggregate, "per_mdp": per_mdp}
