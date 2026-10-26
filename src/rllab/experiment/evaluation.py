"""Distributed policy inference and exact CPU MDP evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

import numpy as np

from rllab.data.exposure import classify_state_exposure
from rllab.experiment.calibration_runtime import CalibrationTable
from rllab.experiment.corpus import (
    ExperimentCorpus,
    InstanceBundle,
    PromptExample,
    prompts_for_rank,
)
from rllab.experiment.modeling import unwrap_policy
from rllab.experiment.runtime import DistributedContext, gather_objects
from rllab.mdp.solve import (
    evaluate_policy,
    mdp_arrays,
    optimal_policy,
    performance_difference,
)


@dataclass(frozen=True)
class PolicySnapshot:
    policies: Mapping[str, np.ndarray]

    def policy(self, mdp_id: str) -> np.ndarray:
        return self.policies[mdp_id]


@dataclass(frozen=True)
class ActionScoreSnapshot:
    """Calibrated four-action scores immediately before masked softmax."""

    scores: Mapping[str, np.ndarray]

    def score(self, mdp_id: str) -> np.ndarray:
        return self.scores[mdp_id]


def policy_snapshot_fingerprint(snapshot: PolicySnapshot) -> str:
    """Hash the exact float64 policy boundary in stable MDP order."""

    digest = sha256()
    for mdp_id in sorted(snapshot.policies):
        values = np.asarray(snapshot.policy(mdp_id), dtype="<f8", order="C")
        digest.update(mdp_id.encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def action_score_snapshot_fingerprint(snapshot: ActionScoreSnapshot) -> str:
    """Hash calibrated FP32 action scores in stable MDP order."""

    digest = sha256()
    for mdp_id in sorted(snapshot.scores):
        values = np.asarray(snapshot.score(mdp_id), dtype="<f4", order="C")
        digest.update(mdp_id.encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def policy_snapshot_l2_difference(
    before: PolicySnapshot, after: PolicySnapshot
) -> float:
    """Return the all-state/action L2 distance between paired policy snapshots."""

    if set(before.policies) != set(after.policies):
        raise ValueError("policy snapshots contain different MDPs")
    squared = 0.0
    for mdp_id in before.policies:
        left = before.policy(mdp_id)
        right = after.policy(mdp_id)
        if left.shape != right.shape:
            raise ValueError(f"policy shapes differ for {mdp_id}")
        squared += float(np.square(right - left).sum(dtype=np.float64))
    return float(np.sqrt(squared))


def policy_snapshot_max_abs_difference(
    left: PolicySnapshot, right: PolicySnapshot
) -> float:
    """Return the largest paired state/action probability difference."""

    if set(left.policies) != set(right.policies):
        raise ValueError("policy snapshots contain different MDPs")
    maximum = 0.0
    for mdp_id in left.policies:
        left_values = left.policy(mdp_id)
        right_values = right.policy(mdp_id)
        if left_values.shape != right_values.shape:
            raise ValueError(f"policy shapes differ for {mdp_id}")
        maximum = max(maximum, float(np.max(np.abs(left_values - right_values))))
    return maximum


def uniform_policy_max_abs_error(
    corpus: ExperimentCorpus, snapshot: PolicySnapshot
) -> float:
    target = 1.0 / len(corpus.action_token_ids)
    return max(
        float(
            np.max(
                np.abs(
                    snapshot.policy(bundle.mdp.mdp_id)[~mdp_arrays(bundle.mdp).terminal]
                    - target
                )
            )
        )
        for bundle in corpus.bundles
    )


def _infer_prompt_rows(
    policy,
    prompts: tuple[PromptExample, ...],
    calibration: CalibrationTable,
    *,
    batch_size: int,
) -> list[tuple[str, str, list[float]]]:
    import torch

    module = unwrap_policy(policy)
    module.eval()
    rows: list[tuple[str, str, list[float]]] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            input_ids = torch.tensor(
                [prompt.input_ids for prompt in batch], dtype=torch.long, device="cuda"
            )
            biases = torch.tensor(
                [calibration.rows[prompt.example_id].fixed_bias for prompt in batch],
                dtype=torch.float32,
                device="cuda",
            )
            probabilities = module.probabilities(input_ids, biases).cpu().numpy()
            for prompt, values in zip(batch, probabilities, strict=True):
                rows.append(
                    (prompt.mdp_id, prompt.state_id, [float(value) for value in values])
                )
    return rows


def _infer_prompt_policy_and_score_rows(
    policy,
    prompts: tuple[PromptExample, ...],
    calibration: CalibrationTable,
    *,
    batch_size: int,
) -> list[tuple[str, str, list[float], list[float]]]:
    import torch

    module = unwrap_policy(policy)
    module.eval()
    rows: list[tuple[str, str, list[float], list[float]]] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            input_ids = torch.tensor(
                [prompt.input_ids for prompt in batch], dtype=torch.long, device="cuda"
            )
            biases = torch.tensor(
                [calibration.rows[prompt.example_id].fixed_bias for prompt in batch],
                dtype=torch.float32,
                device="cuda",
            )
            scores = module(input_ids).float() + biases.float()
            probabilities = torch.log_softmax(scores, dim=-1).exp()
            score_values = scores.cpu().numpy()
            probability_values = probabilities.cpu().numpy()
            for prompt, action_scores, action_probabilities in zip(
                batch, score_values, probability_values, strict=True
            ):
                rows.append(
                    (
                        prompt.mdp_id,
                        prompt.state_id,
                        [float(value) for value in action_scores],
                        [float(value) for value in action_probabilities],
                    )
                )
    return rows


def evaluate_corpus_policy(
    policy,
    corpus: ExperimentCorpus,
    calibration: CalibrationTable,
    *,
    context: DistributedContext,
    batch_size: int,
) -> PolicySnapshot:
    """Shard prompts, not MDPs, so evaluation uses every available device."""

    prompts = prompts_for_rank(
        corpus.all_prompts(), rank=context.rank, world_size=context.world_size
    )
    local_rows = _infer_prompt_rows(policy, prompts, calibration, batch_size=batch_size)
    all_rows = [
        row for rank_rows in gather_objects(local_rows, context) for row in rank_rows
    ]
    return _policy_snapshot_from_rows(corpus, all_rows)


def evaluate_corpus_policy_and_action_scores(
    policy,
    corpus: ExperimentCorpus,
    calibration: CalibrationTable,
    *,
    context: DistributedContext,
    batch_size: int,
) -> tuple[PolicySnapshot, ActionScoreSnapshot]:
    """Evaluate the policy while retaining calibrated pre-softmax scores."""

    prompts = prompts_for_rank(
        corpus.all_prompts(), rank=context.rank, world_size=context.world_size
    )
    local_rows = _infer_prompt_policy_and_score_rows(
        policy, prompts, calibration, batch_size=batch_size
    )
    all_rows = [
        row for rank_rows in gather_objects(local_rows, context) for row in rank_rows
    ]
    policy_rows = [
        (mdp_id, state_id, probabilities)
        for mdp_id, state_id, _, probabilities in all_rows
    ]
    score_rows = [
        (mdp_id, state_id, scores) for mdp_id, state_id, scores, _ in all_rows
    ]
    return (
        _policy_snapshot_from_rows(corpus, policy_rows),
        _action_score_snapshot_from_rows(corpus, score_rows),
    )


def policy_snapshot_from_action_scores(
    corpus: ExperimentCorpus,
    snapshot: ActionScoreSnapshot,
    *,
    context: DistributedContext,
    batch_size: int,
) -> PolicySnapshot:
    """Apply the runtime's FP32 masked-softmax boundary to saved scores."""

    import torch

    arrays = {bundle.mdp.mdp_id: mdp_arrays(bundle.mdp) for bundle in corpus.bundles}
    prompts = prompts_for_rank(
        corpus.all_prompts(), rank=context.rank, world_size=context.world_size
    )
    local_rows: list[tuple[str, str, list[float]]] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            action_scores = torch.tensor(
                [
                    snapshot.score(prompt.mdp_id)[
                        arrays[prompt.mdp_id].state_index[prompt.state_id]
                    ]
                    for prompt in batch
                ],
                dtype=torch.float32,
                device="cuda",
            )
            probabilities = torch.log_softmax(action_scores, dim=-1).exp().cpu().numpy()
            for prompt, values in zip(batch, probabilities, strict=True):
                local_rows.append(
                    (prompt.mdp_id, prompt.state_id, [float(value) for value in values])
                )
    all_rows = [
        row for rank_rows in gather_objects(local_rows, context) for row in rank_rows
    ]
    return _policy_snapshot_from_rows(corpus, all_rows)


def _policy_snapshot_from_rows(
    corpus: ExperimentCorpus,
    all_rows: Iterable[tuple[str, str, list[float]]],
) -> PolicySnapshot:
    all_rows = list(all_rows)
    if len(all_rows) != corpus.example_count:
        raise RuntimeError(
            f"distributed evaluation returned {len(all_rows)} rows, "
            f"expected {corpus.example_count}"
        )
    row_lookup = {(mdp_id, state_id): values for mdp_id, state_id, values in all_rows}
    if len(row_lookup) != len(all_rows):
        raise RuntimeError("distributed evaluation returned duplicate state rows")

    policies: dict[str, np.ndarray] = {}
    for bundle in corpus.bundles:
        arrays = mdp_arrays(bundle.mdp)
        values = np.zeros(
            (len(arrays.state_ids), bundle.mdp.num_actions), dtype=np.float64
        )
        for state_id in bundle.nonterminal_state_ids:
            values[arrays.state_index[state_id]] = np.asarray(
                row_lookup[(bundle.mdp.mdp_id, state_id)], dtype=np.float64
            )
        # Strictly re-normalise in float64 after the FP32 policy boundary.
        nonterminal = ~arrays.terminal
        values[nonterminal] /= values[nonterminal].sum(axis=1, keepdims=True)
        policies[bundle.mdp.mdp_id] = values
    return PolicySnapshot(policies=policies)


def _action_score_snapshot_from_rows(
    corpus: ExperimentCorpus,
    all_rows: Iterable[tuple[str, str, list[float]]],
) -> ActionScoreSnapshot:
    all_rows = list(all_rows)
    if len(all_rows) != corpus.example_count:
        raise RuntimeError(
            f"distributed score evaluation returned {len(all_rows)} rows, "
            f"expected {corpus.example_count}"
        )
    row_lookup = {(mdp_id, state_id): values for mdp_id, state_id, values in all_rows}
    if len(row_lookup) != len(all_rows):
        raise RuntimeError("distributed score evaluation returned duplicate state rows")

    scores: dict[str, np.ndarray] = {}
    for bundle in corpus.bundles:
        arrays = mdp_arrays(bundle.mdp)
        values = np.zeros(
            (len(arrays.state_ids), bundle.mdp.num_actions), dtype=np.float32
        )
        for state_id in bundle.nonterminal_state_ids:
            values[arrays.state_index[state_id]] = np.asarray(
                row_lookup[(bundle.mdp.mdp_id, state_id)], dtype=np.float32
            )
        scores[bundle.mdp.mdp_id] = values
    return ActionScoreSnapshot(scores=scores)


def evaluate_bundle_policy(
    policy,
    bundle: InstanceBundle,
    calibration: CalibrationTable,
    *,
    batch_size: int,
) -> np.ndarray:
    rows = _infer_prompt_rows(
        policy, bundle.ordered_prompts(), calibration, batch_size=batch_size
    )
    arrays = mdp_arrays(bundle.mdp)
    result = np.zeros((len(arrays.state_ids), bundle.mdp.num_actions), dtype=np.float64)
    for _, state_id, probabilities in rows:
        result[arrays.state_index[state_id]] = probabilities
    result[~arrays.terminal] /= result[~arrays.terminal].sum(axis=1, keepdims=True)
    return result


def corpus_objective_summary(
    corpus: ExperimentCorpus, snapshot: PolicySnapshot
) -> dict[str, Any]:
    per_mdp: list[dict[str, Any]] = []
    for bundle in corpus.bundles:
        current = evaluate_policy(bundle.mdp, snapshot.policy(bundle.mdp.mdp_id))
        optimum = evaluate_policy(bundle.mdp, optimal_policy(bundle.mdp))
        gap = optimum.objective - current.objective
        per_mdp.append(
            {
                "mdp_id": bundle.mdp.mdp_id,
                "objective": current.objective,
                "cost": -current.objective,
                "optimal_objective": optimum.objective,
                "optimality_gap": gap,
                "optimality_gap_per_step": gap / bundle.mdp.horizon,
            }
        )
    keys = ("objective", "cost", "optimality_gap", "optimality_gap_per_step")
    aggregate = {
        key: float(np.mean([float(row[key]) for row in per_mdp])) for key in keys
    }
    return {"aggregate": aggregate, "per_mdp": per_mdp}


def state_update_metrics(
    corpus: ExperimentCorpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
    *,
    policy_loss_states: Mapping[str, Iterable[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact state-level update effects with the required exposure partition."""

    rows: list[dict[str, Any]] = []
    max_identity_residual = 0.0
    for bundle in corpus.bundles:
        mdp_id = bundle.mdp.mdp_id
        base_policy = before.policy(mdp_id)
        updated_policy = after.policy(mdp_id)
        arrays = mdp_arrays(bundle.mdp)
        difference = performance_difference(bundle.mdp, base_policy, updated_policy)
        max_identity_residual = max(
            max_identity_residual, float(np.max(np.abs(difference.residual)))
        )
        exposure = {
            record.state_id: record
            for record in classify_state_exposure(
                bundle.mdp, policy_loss_states.get(mdp_id, ())
            )
        }
        for index, state_id in enumerate(arrays.state_ids):
            if arrays.terminal[index]:
                l1_change = 0.0
                max_change = 0.0
                kl = 0.0
            else:
                old = base_policy[index]
                new = updated_policy[index]
                l1_change = float(np.abs(new - old).sum())
                max_change = float(np.abs(new - old).max())
                kl = float(np.sum(old * (np.log(old + 1e-30) - np.log(new + 1e-30))))
            state_exposure = exposure[state_id]
            rows.append(
                {
                    "schema_version": "state-update-metrics/v1",
                    "mdp_id": mdp_id,
                    "state_id": state_id,
                    "depth": int(arrays.depths[index]),
                    "terminal": bool(arrays.terminal[index]),
                    "exposure_class": state_exposure.exposure_class.value,
                    "current_count": state_exposure.current_count,
                    "destination_count": state_exposure.destination_count,
                    "policy_l1": l1_change,
                    "policy_max_abs": max_change,
                    "policy_kl_before_after": kl,
                    "delta_value": float(difference.exact_delta_j[index]),
                    "advantage_projection": float(
                        difference.advantage_projection[index]
                    ),
                    "identity_residual": float(difference.residual[index]),
                }
            )

    exposure_summary: dict[str, Any] = {}
    for exposure_class in (
        "policy_loss_current",
        "destination_context_only",
        "token_absent",
    ):
        selected = [
            row
            for row in rows
            if row["exposure_class"] == exposure_class and not row["terminal"]
        ]
        exposure_summary[exposure_class] = {
            "states": len(selected),
            "mean_policy_l1": (
                float(np.mean([row["policy_l1"] for row in selected]))
                if selected
                else None
            ),
            "mean_delta_value": (
                float(np.mean([row["delta_value"] for row in selected]))
                if selected
                else None
            ),
            "fraction_value_improved": (
                float(np.mean([row["delta_value"] > 0.0 for row in selected]))
                if selected
                else None
            ),
        }
    return rows, {
        "exposure": exposure_summary,
        "max_performance_difference_residual": max_identity_residual,
    }
