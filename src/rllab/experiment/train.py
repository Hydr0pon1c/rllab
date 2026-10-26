"""Controlled full-parameter ORM/exact-PRM training on the vendor PPU stack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from rllab.errors import ArtifactError, RLLabError
from rllab.experiment.calibration_runtime import calibrate_corpus, load_calibration
from rllab.experiment.config import ExperimentConfig
from rllab.experiment.corpus import load_experiment_corpus
from rllab.experiment.evaluation import (
    PolicySnapshot,
    corpus_objective_summary,
    evaluate_corpus_policy,
    policy_snapshot_fingerprint,
    policy_snapshot_l2_difference,
    state_update_metrics,
    uniform_policy_max_abs_error,
)
from rllab.experiment.kernel import directional_kernel_metrics
from rllab.experiment.modeling import (
    load_policy,
    model_snapshot_fingerprint,
    policy_parameter_report,
)
from rllab.experiment.optimization import (
    GRADIENT_SYNCHRONIZATION,
    apply_gradient_fraction,
    build_optimizer,
    optimizer_contract,
    policy_gradient_backward,
    resolve_step_size,
)
from rllab.experiment.rollout import (
    SAMPLER_NAME,
    RolloutBatch,
    assigned_episode_ids,
    collect_rollouts,
    episode_action_uniforms,
    merge_rollout_batches,
    rollout_trajectory_rows,
    trajectory_fingerprint,
)
from rllab.experiment.run_artifacts import (
    RunDirectory,
    clear_active_run,
    latest_checkpoint,
    prepare_run_directory,
    restore_training_state,
    save_checkpoint,
)
from rllab.experiment.runtime import (
    DistributedContext,
    assert_device_capacity,
    barrier,
    gather_objects,
    runtime_report,
    seed_everything,
)


PERFORMANCE_IDENTITY_MAX_RESIDUAL = 1.0e-10


@dataclass(frozen=True)
class TrainingBoundary:
    """Live, restored training state exposed to a purpose-specific observer."""

    config: ExperimentConfig
    policy: Any
    corpus: Any
    calibration: Any
    optimizer: Any
    run: RunDirectory
    context: DistributedContext
    completed_steps: int
    snapshot: PolicySnapshot


BoundaryProbe = Callable[[TrainingBoundary], PolicySnapshot]


def _merge_policy_loss_states(
    destination: dict[str, list[str]], sources: Iterable[Mapping[str, Iterable[str]]]
) -> None:
    for source in sources:
        for mdp_id, state_ids in source.items():
            destination.setdefault(str(mdp_id), []).extend(
                str(state_id) for state_id in state_ids
            )


def _policy_loss_states(
    rollout: RolloutBatch, *, context: DistributedContext
) -> dict[str, list[str]]:
    local: dict[str, list[str]] = {}
    for example in rollout.examples:
        local.setdefault(example.prompt.mdp_id, []).append(example.prompt.state_id)
    result: dict[str, list[str]] = {}
    _merge_policy_loss_states(result, gather_objects(local, context))
    return result


def _sample_budget(
    config: ExperimentConfig, corpus, completed_steps: int
) -> dict[str, int]:
    horizons = {bundle.mdp.horizon for bundle in corpus.bundles}
    if len(horizons) != 1:
        raise RLLabError(
            "one training condition must contain a single fixed horizon; "
            f"found {sorted(horizons)}"
        )
    horizon = next(iter(horizons))
    trajectories_per_update = config.training.trajectories_global
    action_tokens_per_update = trajectories_per_update * horizon
    return {
        "horizon": horizon,
        "trajectories_per_update": trajectories_per_update,
        "action_tokens_per_update": action_tokens_per_update,
        "cumulative_trajectories": completed_steps * trajectories_per_update,
        "cumulative_action_tokens": completed_steps * action_tokens_per_update,
    }


def _write_exact_evaluation(
    run: RunDirectory,
    corpus,
    before: PolicySnapshot,
    after: PolicySnapshot,
    *,
    completed_steps: int,
    sample_budget: Mapping[str, int],
    policy_loss_states: Mapping[str, Iterable[str]],
    reason: str,
) -> dict[str, Any]:
    rows, update_summary = state_update_metrics(
        corpus,
        before,
        after,
        policy_loss_states=policy_loss_states,
    )
    residual = float(update_summary["max_performance_difference_residual"])
    if residual > PERFORMANCE_IDENTITY_MAX_RESIDUAL:
        raise RLLabError(
            f"performance-difference residual {residual} exceeds "
            f"{PERFORMANCE_IDENTITY_MAX_RESIDUAL}"
        )
    relative = f"state_metrics/step-{completed_steps:08d}-{reason}.jsonl.gz"
    run.write_records(relative, rows)
    objective = corpus_objective_summary(corpus, after)
    metric = {
        "schema_version": "rllab-metric/v1",
        "kind": "exact_evaluation",
        "completed_steps": completed_steps,
        "sample_budget": dict(sample_budget),
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "objective": objective,
        "update": update_summary,
        "state_metrics_path": relative,
    }
    run.append_metric(metric)
    return metric


def _local_rollouts(
    config: ExperimentConfig,
    corpus,
    snapshot: PolicySnapshot,
    *,
    update: int,
    context: DistributedContext,
) -> tuple[RolloutBatch, list[dict[str, Any]], int]:
    num_mdps = len(corpus.bundles)
    global_episodes = config.training.trajectories_global
    if global_episodes % num_mdps != 0:
        raise RLLabError(
            "training.trajectories_global must be divisible by the number of MDPs: "
            f"{global_episodes} % {num_mdps} != 0"
        )
    episodes_per_mdp = global_episodes // num_mdps
    local_batches = []
    local_rows: list[dict[str, Any]] = []
    uniform_bin_mismatches = 0
    for mdp_index, bundle in enumerate(corpus.bundles):
        episode_ids = assigned_episode_ids(
            mdp_index=mdp_index,
            num_mdps=num_mdps,
            episodes_per_mdp=episodes_per_mdp,
            rank=context.rank,
            world_size=context.world_size,
        )
        if not episode_ids:
            continue
        uniforms = episode_action_uniforms(
            seed=config.runtime.seed,
            update=update,
            mdp_index=mdp_index,
            episode_ids=episode_ids,
            horizon=bundle.mdp.horizon,
        )
        rollout = collect_rollouts(
            bundle,
            snapshot.policy(bundle.mdp.mdp_id),
            credit_mode=config.training.credit_mode,
            episode_ids=episode_ids,
            action_uniforms=uniforms,
        )
        local_batches.append(rollout)
        rows = rollout_trajectory_rows(
            rollout,
            uniforms,
            seed=config.runtime.seed,
            update=update,
        )
        local_rows.extend(rows)
        if update == 0:
            for row in rows:
                for uniform, step in zip(
                    row["action_uniforms"], row["steps"], strict=True
                ):
                    expected = min(
                        int(float(uniform) * bundle.mdp.num_actions),
                        bundle.mdp.num_actions - 1,
                    )
                    uniform_bin_mismatches += int(int(step["action"]) != expected)

    merged = merge_rollout_batches(
        local_batches,
        credit_mode=config.training.credit_mode,
        shard_name=f"corpus-shard-rank-{context.rank}",
    )
    expected_local = global_episodes // context.world_size
    if merged.episodes != expected_local:
        raise RLLabError(
            f"rank {context.rank} has {merged.episodes} episodes, "
            f"expected {expected_local}"
        )
    return merged, local_rows, uniform_bin_mismatches


def _global_rollout_summary(
    rollout: RolloutBatch,
    trajectory_rows: list[dict[str, Any]],
    uniform_bin_mismatches: int,
    *,
    expected_episodes: int,
    context: DistributedContext,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = {
        "returns": list(rollout.total_returns),
        "credits": [example.credit for example in rollout.examples],
        "probabilities": [example.action_probability for example in rollout.examples],
        "trajectory_rows": trajectory_rows,
        "uniform_bin_mismatches": int(uniform_bin_mismatches),
    }
    gathered = gather_objects(payload, context)
    returns = np.asarray(
        [value for item in gathered for value in item["returns"]], dtype=np.float64
    )
    credits = np.asarray(
        [value for item in gathered for value in item["credits"]], dtype=np.float64
    )
    probabilities = np.asarray(
        [value for item in gathered for value in item["probabilities"]],
        dtype=np.float64,
    )
    rows = [row for item in gathered for row in item["trajectory_rows"]]
    rows.sort(key=lambda row: (str(row["mdp_id"]), int(row["episode_id"])))
    if returns.size != expected_episodes or len(rows) != expected_episodes:
        raise RLLabError(
            "global rollout is missing or duplicating episodes: "
            f"returns={returns.size}, rows={len(rows)}, expected={expected_episodes}"
        )
    mismatch_count = sum(int(item["uniform_bin_mismatches"]) for item in gathered)
    return (
        {
            "episodes_global": int(returns.size),
            "states_global": int(credits.size),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "mean_credit": float(credits.mean()),
            "std_credit": float(credits.std()),
            "mean_abs_credit": float(np.abs(credits).mean()),
            "mean_sampled_action_probability": float(probabilities.mean()),
            "max_sampled_action_probability_deviation_from_uniform": float(
                np.max(np.abs(probabilities - 0.25))
            ),
            "trajectory_fingerprint": trajectory_fingerprint(rows),
            "initial_uniform_bin_mismatches": mismatch_count,
        },
        rows,
    )


def _scheduled_evaluation(config: ExperimentConfig, completed_steps: int) -> bool:
    if config.training.evaluation_steps:
        return completed_steps in config.training.evaluation_steps
    return (
        completed_steps % config.training.evaluation_interval == 0
        or completed_steps == config.training.steps
    )


def run_training(
    config: ExperimentConfig,
    *,
    context: DistributedContext,
    allow_dirty: bool = False,
    boundary_probe_steps: tuple[int, ...] = (),
    boundary_probe: BoundaryProbe | None = None,
) -> dict[str, Any]:
    import torch

    config.validate()
    if bool(boundary_probe_steps) != (boundary_probe is not None):
        raise ValueError(
            "boundary_probe_steps and boundary_probe must be provided together"
        )
    if tuple(sorted(set(boundary_probe_steps))) != boundary_probe_steps or any(
        step < 1 or step > config.training.steps for step in boundary_probe_steps
    ):
        raise ValueError(
            "boundary_probe_steps must be sorted, unique, and inside training"
        )
    if config.training.trajectories_global % context.world_size != 0:
        raise RLLabError(
            "training.trajectories_global must be divisible by WORLD_SIZE: "
            f"{config.training.trajectories_global} % {context.world_size} != 0"
        )
    seed_everything(
        config.runtime.seed,
        context,
        deterministic=config.runtime.deterministic_algorithms,
    )
    capacity = assert_device_capacity(config.runtime.min_free_device_gib, context)
    local_runtime = runtime_report(context)
    local_runtime["initial_capacity"] = capacity
    runtime_reports = gather_objects(local_runtime, context)

    corpus = load_experiment_corpus(
        config.data.path,
        split=config.data.split,
        labeling_mode=config.data.labeling_mode,
        geometry_path=config.data.geometry_path,
        max_instances=config.data.max_instances,
    )
    base_model_fingerprint = model_snapshot_fingerprint(
        config.model.path, config.model.revision
    )
    run = prepare_run_directory(
        config,
        corpus,
        context=context,
        model_fingerprint=base_model_fingerprint,
        runtime_reports=runtime_reports,
        allow_dirty=allow_dirty,
    )
    checkpoint = latest_checkpoint(run) if config.output.resume else None
    if config.output.resume and checkpoint is None:
        raise ArtifactError(f"--resume requested but {run.path} has no checkpoint")
    model_config = (
        replace(config.model, path=str(checkpoint / "model"))
        if checkpoint is not None
        else config.model
    )

    policy = load_policy(model_config, corpus.action_token_ids)
    parameter_report = policy_parameter_report(policy)
    optimizer = build_optimizer(policy, learning_rate=config.training.learning_rate)
    optimizer_report = optimizer_contract(optimizer)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.runtime.seed + 10_000_019 * context.rank)
    completed_steps = 0
    if checkpoint is not None:
        completed_steps = restore_training_state(
            checkpoint,
            optimizer,
            generator=generator,
            context=context,
        )
    if completed_steps >= config.training.steps:
        raise ArtifactError(
            f"checkpoint already completed {completed_steps} steps, target is "
            f"{config.training.steps}"
        )

    if run.calibration_path.is_file():
        calibration = load_calibration(
            run.calibration_path,
            corpus=corpus,
            model_fingerprint=base_model_fingerprint,
        )
    elif checkpoint is not None:
        raise ArtifactError(
            "resume checkpoint exists but initial calibration is missing"
        )
    else:
        calibration = calibrate_corpus(
            policy,
            corpus,
            context=context,
            model_fingerprint=base_model_fingerprint,
            batch_size=config.training.forward_batch_states,
            output_path=run.calibration_path,
        )

    current_snapshot = evaluate_corpus_policy(
        policy,
        corpus,
        calibration,
        context=context,
        batch_size=config.training.forward_batch_states,
    )
    if completed_steps == 0:
        uniform_error = uniform_policy_max_abs_error(corpus, current_snapshot)
        if uniform_error > 2.0e-6:
            raise RLLabError(
                f"fixed calibration failed at runtime: max uniform error={uniform_error}"
            )
    else:
        uniform_error = None

    run_start_objective = corpus_objective_summary(corpus, current_snapshot)
    if context.is_main:
        run.update_status(
            "running",
            completed_steps=completed_steps,
            target_steps=config.training.steps,
        )
        run.append_metric(
            {
                "schema_version": "rllab-metric/v1",
                "kind": "launch",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "completed_steps": completed_steps,
                "world_size": context.world_size,
                "parameter_report": parameter_report,
                "optimizer_contract": optimizer_report,
                "gradient_synchronization": GRADIENT_SYNCHRONIZATION,
                "sampler": SAMPLER_NAME,
                "scale_rule": config.training.scale_rule,
                "target_parameter_step_l2": (config.training.target_parameter_step_l2),
                "evaluation_steps": list(config.training.evaluation_steps),
                "boundary_probe_steps": list(boundary_probe_steps),
                "sample_budget": _sample_budget(config, corpus, completed_steps),
                "initial_uniform_max_abs_error": uniform_error,
                "initial_policy_fingerprint": policy_snapshot_fingerprint(
                    current_snapshot
                ),
                "initial_objective": run_start_objective,
            }
        )

    last_evaluation = current_snapshot
    interval_states: dict[str, list[str]] = {}
    last_metric: dict[str, Any] | None = None
    cumulative_policy_response_l2 = 0.0
    training_started = time.perf_counter()

    for step in range(completed_steps, config.training.steps):
        step_started = time.perf_counter()
        before_snapshot = current_snapshot
        boundary_fingerprint = policy_snapshot_fingerprint(before_snapshot)
        rollout, local_trajectory_rows, local_mismatches = _local_rollouts(
            config,
            corpus,
            before_snapshot,
            update=step,
            context=context,
        )
        backward = policy_gradient_backward(
            policy,
            rollout,
            calibration,
            optimizer=optimizer,
            microbatch_states=config.training.train_microbatch_states,
            context=context,
        )
        current_update_states = _policy_loss_states(rollout, context=context)
        _merge_policy_loss_states(interval_states, (current_update_states,))
        rollout_summary, global_trajectory_rows = _global_rollout_summary(
            rollout,
            local_trajectory_rows,
            local_mismatches,
            expected_episodes=config.training.trajectories_global,
            context=context,
        )
        if step == 0 and rollout_summary["initial_uniform_bin_mismatches"] != 0:
            raise RLLabError(
                "initial evaluated-policy sampler disagrees with exact uniform bins: "
                f"{rollout_summary['initial_uniform_bin_mismatches']} mismatches"
            )

        effective_lr, parameter_step_l2 = resolve_step_size(
            scale_rule=config.training.scale_rule,
            learning_rate=config.training.learning_rate,
            target_parameter_step_l2=config.training.target_parameter_step_l2,
            gradient_l2=backward.gradient_l2,
        )
        is_kernel_probe = step in config.training.kernel_probe_steps
        if is_kernel_probe:
            apply_gradient_fraction(
                policy,
                step_size=effective_lr * config.training.kernel_probe_fraction,
            )
            fractional_snapshot = evaluate_corpus_policy(
                policy,
                corpus,
                calibration,
                context=context,
                batch_size=config.training.forward_batch_states,
            )
            apply_gradient_fraction(
                policy,
                step_size=effective_lr * (1.0 - config.training.kernel_probe_fraction),
            )
        else:
            fractional_snapshot = None
            apply_gradient_fraction(policy, step_size=effective_lr)

        # The full update is now materialized. Release the ~30 GiB FP32
        # gradient set before the all-state inference boundary; kernel probes
        # retain it only until the fractional and remaining steps are applied.
        optimizer.zero_grad(set_to_none=True)
        after_snapshot = evaluate_corpus_policy(
            policy,
            corpus,
            calibration,
            context=context,
            batch_size=config.training.forward_batch_states,
        )
        if is_kernel_probe:
            assert fractional_snapshot is not None
            kernel_rows, kernel_summary = directional_kernel_metrics(
                corpus,
                before_snapshot,
                fractional_snapshot,
                after_snapshot,
                probe_fraction=config.training.kernel_probe_fraction,
                policy_loss_states=current_update_states,
            )
            correlation = kernel_summary["policy_delta_correlation"]
            finite_correlation = correlation is not None and math.isfinite(
                float(correlation)
            )
            if context.is_main:
                kernel_relative = f"kernel/step-{step + 1:08d}.jsonl.gz"
                run.write_records(kernel_relative, kernel_rows)
                run.append_metric(
                    {
                        "schema_version": "rllab-metric/v1",
                        "kind": "directional_kernel",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "completed_steps": step + 1,
                        "sample_budget": _sample_budget(config, corpus, step + 1),
                        "summary": kernel_summary,
                        "kernel_path": kernel_relative,
                    }
                )
            if not finite_correlation:
                # Let rank 0 finish the diagnostic artifacts before every rank
                # exits. A non-finite response cannot support kernel analysis.
                barrier(context)
                raise RLLabError(
                    f"kernel correlation at update {step + 1} is not finite: "
                    f"{correlation}"
                )

        response_l2 = policy_snapshot_l2_difference(before_snapshot, after_snapshot)
        cumulative_policy_response_l2 += response_l2
        before_objective = corpus_objective_summary(corpus, before_snapshot)
        after_objective = corpus_objective_summary(corpus, after_snapshot)
        objective_gain = float(
            after_objective["aggregate"]["objective"]
            - before_objective["aggregate"]["objective"]
        )
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started

        if context.is_main:
            trajectory_relative = f"trajectories/step-{step + 1:08d}.jsonl.gz"
            run.write_records(trajectory_relative, global_trajectory_rows)
            run.append_metric(
                {
                    "schema_version": "rllab-metric/v1",
                    "kind": "update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "completed_steps": step + 1,
                    "credit_mode": config.training.credit_mode,
                    "loss": backward.global_loss,
                    "gradient_l2": backward.gradient_l2,
                    "scale_rule": config.training.scale_rule,
                    "effective_learning_rate": effective_lr,
                    "parameter_step_l2": parameter_step_l2,
                    "target_parameter_step_l2": (
                        config.training.target_parameter_step_l2
                    ),
                    "policy_boundary_fingerprint": boundary_fingerprint,
                    "policy_response_l2": response_l2,
                    "cumulative_policy_response_l2": cumulative_policy_response_l2,
                    "objective_before": before_objective["aggregate"]["objective"],
                    "objective_after": after_objective["aggregate"]["objective"],
                    "objective_gain": objective_gain,
                    "objective_gain_from_run_start": float(
                        after_objective["aggregate"]["objective"]
                        - run_start_objective["aggregate"]["objective"]
                    ),
                    "objective_gain_per_cumulative_policy_response_l2": (
                        float(
                            after_objective["aggregate"]["objective"]
                            - run_start_objective["aggregate"]["objective"]
                        )
                        / cumulative_policy_response_l2
                        if cumulative_policy_response_l2 > 0.0
                        else None
                    ),
                    "step_seconds": step_seconds,
                    "states_per_second_global": rollout_summary["states_global"]
                    / step_seconds,
                    "sample_budget": _sample_budget(config, corpus, step + 1),
                    "rollout": rollout_summary,
                    "trajectory_path": trajectory_relative,
                    "sampler": SAMPLER_NAME,
                    "device_memory": {
                        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    },
                }
            )

        should_evaluate = _scheduled_evaluation(config, step + 1) or is_kernel_probe
        if should_evaluate:
            if context.is_main:
                last_metric = _write_exact_evaluation(
                    run,
                    corpus,
                    last_evaluation,
                    after_snapshot,
                    completed_steps=step + 1,
                    sample_budget=_sample_budget(config, corpus, step + 1),
                    policy_loss_states=interval_states,
                    reason="kernel-update" if is_kernel_probe else "scheduled",
                )
            last_evaluation = after_snapshot
            interval_states = {}

        if step + 1 in boundary_probe_steps:
            assert boundary_probe is not None
            expected_fingerprint = policy_snapshot_fingerprint(after_snapshot)
            restored_snapshot = boundary_probe(
                TrainingBoundary(
                    config=config,
                    policy=policy,
                    corpus=corpus,
                    calibration=calibration,
                    optimizer=optimizer,
                    run=run,
                    context=context,
                    completed_steps=step + 1,
                    snapshot=after_snapshot,
                )
            )
            restored_fingerprint = policy_snapshot_fingerprint(restored_snapshot)
            if restored_fingerprint != expected_fingerprint:
                raise RLLabError(
                    "boundary probe changed the training policy: "
                    f"expected={expected_fingerprint}, restored={restored_fingerprint}"
                )
            after_snapshot = restored_snapshot

        if context.is_main:
            run.update_status(
                "running",
                completed_steps=step + 1,
                target_steps=config.training.steps,
            )

        should_checkpoint = (
            (step + 1) % config.training.checkpoint_interval == 0
            or config.training.save_final_checkpoint
            and step + 1 == config.training.steps
        )
        if should_checkpoint:
            save_checkpoint(
                run,
                policy,
                optimizer,
                completed_steps=step + 1,
                generator=generator,
                context=context,
                keep=config.training.keep_checkpoints,
            )
        if context.is_main and (
            (step + 1) % config.runtime.log_every == 0
            or step + 1 == config.training.steps
        ):
            print(
                json.dumps(
                    {
                        "rllab": "training_progress",
                        "run_id": run.run_id,
                        "completed_steps": step + 1,
                        "target_steps": config.training.steps,
                        "objective_gain_from_run_start": float(
                            after_objective["aggregate"]["objective"]
                            - run_start_objective["aggregate"]["objective"]
                        ),
                        "step_seconds": step_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        current_snapshot = after_snapshot

    total_seconds = time.perf_counter() - training_started
    final_summary = corpus_objective_summary(corpus, current_snapshot)
    final_policy_fingerprint = policy_snapshot_fingerprint(current_snapshot)
    if context.is_main:
        run.update_status(
            "completed",
            completed_steps=config.training.steps,
            target_steps=config.training.steps,
            total_training_seconds=total_seconds,
            cumulative_policy_response_l2=cumulative_policy_response_l2,
            final_objective=final_summary,
            final_policy_fingerprint=final_policy_fingerprint,
        )
    barrier(context)
    clear_active_run(run)
    return {
        "run_id": run.run_id,
        "run_path": str(run.path),
        "completed_steps": config.training.steps,
        "world_size": context.world_size,
        "total_training_seconds": total_seconds,
        "cumulative_policy_response_l2": cumulative_policy_response_l2,
        "final_objective": final_summary,
        "final_policy_fingerprint": final_policy_fingerprint,
        "last_evaluation": last_metric,
    }
