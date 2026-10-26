"""Deterministic on-policy rollouts from an already evaluated policy boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from rllab.artifacts import canonical_json_bytes
from rllab.experiment.corpus import InstanceBundle, PromptExample
from rllab.mdp.solve import evaluate_policy, mdp_arrays


SAMPLER_NAME = "episode-indexed-evaluated-policy-v2"
_SAMPLER_NAMESPACE = 0x524C4C42
_DYNAMIC_NAMESPACE = 0x524C4C44


@dataclass(frozen=True)
class UpdateExample:
    episode: int
    step: int
    prompt: PromptExample
    action: int
    credit: float
    action_probability: float
    immediate_cost: int
    target_state_id: str


@dataclass(frozen=True)
class RolloutBatch:
    mdp_id: str
    credit_mode: str
    episodes: int
    episode_ids: tuple[int, ...]
    examples: tuple[UpdateExample, ...]
    total_returns: tuple[float, ...]
    current_state_counts: Mapping[str, int]
    statistics: Mapping[str, float]


def assigned_episode_ids(
    *,
    mdp_index: int,
    num_mdps: int,
    episodes_per_mdp: int,
    rank: int,
    world_size: int,
) -> tuple[int, ...]:
    """Assign episode-major global IDs evenly and independently of placement."""

    if not 0 <= mdp_index < num_mdps:
        raise ValueError("mdp_index falls outside num_mdps")
    if episodes_per_mdp < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid episode assignment dimensions")
    return tuple(
        episode
        for episode in range(episodes_per_mdp)
        if (episode * num_mdps + mdp_index) % world_size == rank
    )


def episode_action_uniforms(
    *,
    seed: int,
    update: int,
    mdp_index: int,
    episode_ids: Sequence[int],
    horizon: int,
) -> np.ndarray:
    """Return counter-keyed uniforms shared by paired credit conditions.

    Update zero intentionally preserves the D04 sampler entropy so E05 exactly
    reuses its audited initial B16 trajectory table. Later updates use a new
    namespace and include the update index explicitly.
    """

    if seed < 0 or update < 0 or mdp_index < 0 or horizon < 1:
        raise ValueError("invalid episode-uniform key")
    rows = []
    for episode_id in episode_ids:
        if episode_id < 0:
            raise ValueError("episode IDs must be non-negative")
        entropy = (
            [_SAMPLER_NAMESPACE, int(seed), int(mdp_index), int(episode_id)]
            if update == 0
            else [
                _DYNAMIC_NAMESPACE,
                int(seed),
                int(update),
                int(mdp_index),
                int(episode_id),
            ]
        )
        sequence = np.random.SeedSequence(entropy)
        rows.append(np.random.default_rng(sequence).random(horizon, dtype=np.float64))
    if not rows:
        return np.empty((0, horizon), dtype=np.float64)
    return np.stack(rows, axis=0)


def inverse_cdf_actions(probabilities: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Sample categorical rows from explicit uniforms with auditable boundaries."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, actions]")
    if uniforms.ndim != 1 or uniforms.shape[0] != probabilities.shape[0]:
        raise ValueError("uniforms must have shape [batch]")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("probability rows must sum to one")
    if not np.all(np.isfinite(uniforms)) or not np.all(
        (uniforms >= 0.0) & (uniforms < 1.0)
    ):
        raise ValueError("uniforms must be finite values in [0, 1)")
    cumulative = np.cumsum(probabilities, axis=-1)
    actions = (uniforms[:, None] >= cumulative).sum(axis=-1)
    return np.minimum(actions, probabilities.shape[1] - 1).astype(np.int64)


def _rollout_statistics(
    examples: Sequence[UpdateExample], total_returns: Sequence[float]
) -> dict[str, float]:
    credits = np.asarray([example.credit for example in examples], dtype=np.float64)
    returns = np.asarray(total_returns, dtype=np.float64)
    probabilities = np.asarray(
        [example.action_probability for example in examples], dtype=np.float64
    )
    if credits.size == 0 or returns.size == 0:
        raise ValueError("rollout statistics require non-empty episodes")
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_credit": float(credits.mean()),
        "std_credit": float(credits.std()),
        "mean_abs_credit": float(np.abs(credits).mean()),
        "mean_sampled_action_probability": float(probabilities.mean()),
    }


def collect_rollouts(
    bundle: InstanceBundle,
    reference_policy: np.ndarray,
    *,
    credit_mode: str,
    episode_ids: Sequence[int],
    action_uniforms: np.ndarray,
) -> RolloutBatch:
    """Sample from the exact policy array used for both credit and evaluation."""

    if credit_mode not in {"orm", "exact_prm"}:
        raise ValueError(f"unsupported credit mode: {credit_mode}")
    episode_ids = tuple(int(value) for value in episode_ids)
    if not episode_ids or tuple(sorted(set(episode_ids))) != episode_ids:
        raise ValueError("episode_ids must be non-empty, sorted, and unique")
    mdp = bundle.mdp
    action_uniforms = np.asarray(action_uniforms, dtype=np.float64)
    if action_uniforms.shape != (len(episode_ids), mdp.horizon):
        raise ValueError(
            "action_uniforms must have shape "
            f"({len(episode_ids)}, {mdp.horizon}), got {action_uniforms.shape}"
        )
    if not np.all(np.isfinite(action_uniforms)) or not np.all(
        (action_uniforms >= 0.0) & (action_uniforms < 1.0)
    ):
        raise ValueError("action_uniforms must be finite values in [0, 1)")

    arrays = mdp_arrays(mdp)
    reference_policy = np.asarray(reference_policy, dtype=np.float64)
    expected_shape = (len(arrays.state_ids), mdp.num_actions)
    if reference_policy.shape != expected_shape:
        raise ValueError(
            f"reference policy shape is {reference_policy.shape}, expected {expected_shape}"
        )
    evaluation = evaluate_policy(mdp, reference_policy)
    outgoing = mdp.outgoing()
    state_by_id = {state.state_id: state for state in mdp.states}

    current_states = [mdp.start_state for _ in episode_ids]
    episode_steps: list[list[dict[str, Any]]] = [[] for _ in episode_ids]
    transition_costs = [0 for _ in episode_ids]
    state_counts: dict[str, int] = {}

    for step in range(mdp.horizon):
        state_indices = np.asarray(
            [arrays.state_index[state_id] for state_id in current_states],
            dtype=np.int64,
        )
        probabilities = reference_policy[state_indices]
        actions = inverse_cdf_actions(probabilities, action_uniforms[:, step])
        sampled_probabilities = probabilities[np.arange(len(episode_ids)), actions]
        next_states: list[str] = []
        for dense_episode, (state_id, action, action_probability) in enumerate(
            zip(current_states, actions, sampled_probabilities, strict=True)
        ):
            transition = outgoing[state_id][int(action)]
            episode_steps[dense_episode].append(
                {
                    "step": step,
                    "state_id": state_id,
                    "prompt": bundle.prompts[state_id],
                    "action": int(action),
                    "action_probability": float(action_probability),
                    "immediate_cost": int(transition.cost),
                    "target_state_id": transition.target,
                }
            )
            transition_costs[dense_episode] += int(transition.cost)
            state_counts[state_id] = state_counts.get(state_id, 0) + 1
            next_states.append(transition.target)
        current_states = next_states

    if not all(state_by_id[state_id].terminal for state_id in current_states):
        raise RuntimeError("fixed-horizon rollout did not terminate")
    total_costs = [
        transition_cost + state_by_id[terminal].terminal_cost
        for transition_cost, terminal in zip(
            transition_costs, current_states, strict=True
        )
    ]
    total_returns = tuple(-float(cost) for cost in total_costs)

    examples: list[UpdateExample] = []
    for dense_episode, (steps, outcome) in enumerate(
        zip(episode_steps, total_returns, strict=True)
    ):
        for record in steps:
            state_index = arrays.state_index[record["state_id"]]
            credit = (
                outcome - evaluation.objective
                if credit_mode == "orm"
                else float(evaluation.advantage[state_index, int(record["action"])])
            )
            examples.append(
                UpdateExample(
                    episode=dense_episode,
                    step=int(record["step"]),
                    prompt=record["prompt"],
                    action=int(record["action"]),
                    credit=float(credit),
                    action_probability=float(record["action_probability"]),
                    immediate_cost=int(record["immediate_cost"]),
                    target_state_id=str(record["target_state_id"]),
                )
            )

    return RolloutBatch(
        mdp_id=mdp.mdp_id,
        credit_mode=credit_mode,
        episodes=len(episode_ids),
        episode_ids=episode_ids,
        examples=tuple(examples),
        total_returns=total_returns,
        current_state_counts=state_counts,
        statistics=_rollout_statistics(examples, total_returns),
    )


def merge_rollout_batches(
    rollouts: Sequence[RolloutBatch], *, credit_mode: str, shard_name: str
) -> RolloutBatch:
    """Merge multiple MDP-local batches for one distributed rank."""

    if not rollouts or any(rollout.credit_mode != credit_mode for rollout in rollouts):
        raise ValueError("rollout merge requires non-empty matching credit batches")
    examples = []
    returns = []
    state_counts: dict[str, int] = {}
    offset = 0
    for rollout in rollouts:
        examples.extend(
            replace(example, episode=example.episode + offset)
            for example in rollout.examples
        )
        returns.extend(rollout.total_returns)
        offset += rollout.episodes
        for state_id, count in rollout.current_state_counts.items():
            state_counts[state_id] = state_counts.get(state_id, 0) + int(count)
    return RolloutBatch(
        mdp_id=shard_name,
        credit_mode=credit_mode,
        episodes=len(returns),
        episode_ids=tuple(range(len(returns))),
        examples=tuple(examples),
        total_returns=tuple(float(value) for value in returns),
        current_state_counts=state_counts,
        statistics=_rollout_statistics(examples, returns),
    )


def rollout_trajectory_rows(
    rollout: RolloutBatch,
    action_uniforms: np.ndarray,
    *,
    seed: int,
    update: int,
) -> list[dict[str, Any]]:
    """Serialize one MDP-local trajectory table with stable episode IDs."""

    action_uniforms = np.asarray(action_uniforms, dtype=np.float64)
    if action_uniforms.ndim != 2 or action_uniforms.shape[0] != rollout.episodes:
        raise ValueError(
            "action_uniforms must have one two-dimensional row per episode"
        )
    examples_by_episode: dict[int, list[UpdateExample]] = {
        index: [] for index in range(rollout.episodes)
    }
    for example in rollout.examples:
        if example.episode not in examples_by_episode:
            raise ValueError("rollout example refers to an unknown dense episode")
        examples_by_episode[example.episode].append(example)
    rows = []
    for dense_episode, episode_id in enumerate(rollout.episode_ids):
        examples = sorted(
            examples_by_episode[dense_episode], key=lambda example: example.step
        )
        if len(examples) != action_uniforms.shape[1] or [
            example.step for example in examples
        ] != list(range(action_uniforms.shape[1])):
            raise ValueError("trajectory steps do not match the action-uniform horizon")
        rows.append(
            {
                "schema_version": "rllab-training-trajectory/v1",
                "seed": int(seed),
                "update": int(update),
                "mdp_id": rollout.mdp_id,
                "episode_id": int(episode_id),
                "total_return": float(rollout.total_returns[dense_episode]),
                "action_uniforms": [
                    float(value) for value in action_uniforms[dense_episode]
                ],
                "steps": [
                    {
                        "step": example.step,
                        "state_id": example.prompt.state_id,
                        "action": example.action,
                        "action_probability": example.action_probability,
                        "immediate_cost": example.immediate_cost,
                        "target_state_id": example.target_state_id,
                    }
                    for example in examples
                ],
            }
        )
    return rows


def trajectory_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            int(row["update"]),
            str(row["mdp_id"]),
            int(row["episode_id"]),
        ),
    )
    return sha256(canonical_json_bytes(ordered)).hexdigest()
