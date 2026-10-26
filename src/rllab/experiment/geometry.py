"""Measured frozen-model state-token geometry and paired relabel artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from rllab.artifacts import (
    JsonlGzipWriter,
    canonical_json_bytes,
    sha256_file,
    source_tree_fingerprint,
)
from rllab.data.calibration import uniformizing_fixed_bias
from rllab.data.prompts import build_all_prompt_records
from rllab.data.relabel import make_score_aligned_labeling
from rllab.data.tokens import TokenRegistry, load_tokenizer
from rllab.errors import ArtifactError, RLLabError
from rllab.experiment.calibration_runtime import calibrate_corpus
from rllab.experiment.config import ExperimentConfig
from rllab.experiment.corpus import (
    ExperimentCorpus,
    InstanceBundle,
    load_experiment_corpus,
)
from rllab.experiment.modeling import (
    load_policy,
    model_snapshot_fingerprint,
    unwrap_policy,
)
from rllab.experiment.optimization import (
    apply_gradient_fraction,
    exact_uniform_policy_gradient_backward,
)
from rllab.experiment.run_artifacts import atomic_json, git_provenance
from rllab.experiment.runtime import (
    DistributedContext,
    barrier,
    broadcast_object,
    gather_objects,
    runtime_report,
)
from rllab.storage import repository_root, storage_lock


GEOMETRY_SCHEMA_VERSION = "rllab-geometry/v1"
ASSIGNMENT_SCOPES = {"full", "depth"}
AssignmentBlocks = tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]


@dataclass(frozen=True)
class ProbeMetadata:
    mdp_index: int
    state_index: int
    token_index: int
    source_state_id: str
    occupancy: float
    advantage: tuple[float, ...]


@dataclass(frozen=True)
class ProbeExample:
    metadata: ProbeMetadata
    input_ids: tuple[int, ...]


def _geometry_identity(
    config: ExperimentConfig,
    corpus: ExperimentCorpus,
    model_fingerprint: str,
    assignment_scope: str,
) -> dict[str, Any]:
    method = (
        "isolated-state-slot-directional-transfer/v1"
        if assignment_scope == "full"
        else "isolated-state-slot-directional-transfer/v2-depth-block"
    )
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "method": method,
        "assignment_scope": assignment_scope,
        "base_dataset_build_id": corpus.identity.dataset_build_id,
        "token_registry_fingerprint": corpus.identity.token_registry_fingerprint,
        "model_fingerprint": model_fingerprint,
        "model_revision": config.model.revision,
        "attention_implementation": config.model.attention_implementation,
        "compute_dtype": config.model.compute_dtype,
        "reference_gradient": "exact-root-policy-gradient-under-uniform-policy",
        "probe_step_size": config.training.learning_rate,
        "split": corpus.split,
        "instances": [bundle.mdp.mdp_id for bundle in corpus.bundles],
        "code_fingerprint": source_tree_fingerprint(repository_root()),
    }


def _build_id(identity: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(dict(identity))).hexdigest()[:16]


def _assignment_blocks(
    bundle: InstanceBundle,
    registry: TokenRegistry,
    *,
    assignment_scope: str,
) -> AssignmentBlocks | None:
    if assignment_scope == "full":
        return None
    if assignment_scope != "depth":
        raise RLLabError(f"unsupported geometry assignment scope: {assignment_scope}")

    states = bundle.mdp.ordered_states()
    candidate_column = {
        token.token_id: column
        for column, token in enumerate(registry.state_tokens[: len(states)])
    }
    blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for depth in sorted({state.depth for state in states}):
        rows = tuple(
            index for index, state in enumerate(states) if state.depth == depth
        )
        try:
            columns = tuple(
                sorted(
                    candidate_column[
                        bundle.labeling.state_token_ids[states[index].state_id]
                    ]
                    for index in rows
                )
            )
        except KeyError as exc:
            raise ArtifactError(
                f"random labeling uses a token outside the candidate pool: {exc}"
            ) from exc
        if len(set(columns)) != len(rows):
            raise ArtifactError(
                f"depth {depth} random token pool is not one-to-one for "
                f"{bundle.mdp.mdp_id}"
            )
        blocks.append((rows, columns))
    return tuple(blocks)


def _candidate_columns_by_state(
    state_count: int,
    blocks: AssignmentBlocks | None,
) -> tuple[tuple[int, ...], ...]:
    if blocks is None:
        columns = tuple(range(state_count))
        return tuple(columns for _ in range(state_count))
    result: list[tuple[int, ...] | None] = [None] * state_count
    for rows, columns in blocks:
        for row in rows:
            if result[row] is not None:
                raise ArtifactError(f"geometry assignment row {row} appears twice")
            result[row] = columns
    if any(columns is None for columns in result):
        raise ArtifactError("geometry assignment blocks do not cover every state")
    return tuple(columns for columns in result if columns is not None)


def _geometry_workload(
    corpus: ExperimentCorpus,
    registry: TokenRegistry,
    *,
    assignment_scope: str,
) -> dict[str, Any]:
    by_mdp: dict[str, Any] = {}
    total_examples = 0
    total_pairs = 0
    for bundle in corpus.bundles:
        states = bundle.mdp.ordered_states()
        state_index = {state.state_id: index for index, state in enumerate(states)}
        blocks = _assignment_blocks(bundle, registry, assignment_scope=assignment_scope)
        candidates = _candidate_columns_by_state(len(states), blocks)
        examples = sum(
            len(candidates[state_index[canonical_id]])
            for prompt in bundle.ordered_prompts()
            for _position, _role, canonical_id in prompt.state_slots
        )
        block_sizes = (
            [len(states)] if blocks is None else [len(rows) for rows, _ in blocks]
        )
        pairs = sum(size * size for size in block_sizes)
        by_mdp[bundle.mdp.mdp_id] = {
            "states": len(states),
            "block_sizes": block_sizes,
            "candidate_pairs": pairs,
            "probe_examples_per_pass": examples,
        }
        total_examples += examples
        total_pairs += pairs
    return {
        "assignment_scope": assignment_scope,
        "candidate_pairs": total_pairs,
        "probe_examples_per_pass": total_examples,
        "probe_examples_two_passes": 2 * total_examples,
        "by_mdp": by_mdp,
    }


def _probe_examples(
    corpus: ExperimentCorpus,
    registry: TokenRegistry,
    *,
    rank: int,
    world_size: int,
    assignment_scope: str,
) -> Iterator[ProbeExample]:
    global_index = 0
    for mdp_index, bundle in enumerate(corpus.bundles):
        states = bundle.mdp.ordered_states()
        state_count = len(states)
        candidates = registry.state_tokens[:state_count]
        blocks = _assignment_blocks(bundle, registry, assignment_scope=assignment_scope)
        candidate_columns = _candidate_columns_by_state(state_count, blocks)
        neutral = registry.state_tokens[state_count : state_count + 8]
        if len(neutral) < 5:
            raise ArtifactError(
                "token registry needs at least five spare state tokens for isolated probes"
            )
        ordered_prompts = bundle.ordered_prompts()
        for state_index, state in enumerate(states):
            affected = [
                prompt
                for prompt in ordered_prompts
                if any(slot[2] == state.state_id for slot in prompt.state_slots)
            ]
            if not affected:
                raise ArtifactError(
                    f"state {bundle.mdp.mdp_id}:{state.state_id} has no prompt occurrence"
                )
            for token_index in candidate_columns[state_index]:
                candidate = candidates[token_index]
                for prompt in affected:
                    if global_index % world_size == rank:
                        values = list(prompt.input_ids)
                        for slot_index, (position, _role, canonical_id) in enumerate(
                            prompt.state_slots
                        ):
                            values[position] = (
                                candidate.token_id
                                if canonical_id == state.state_id
                                else neutral[slot_index].token_id
                            )
                        truth = bundle.truth[prompt.state_id]
                        advantage = tuple(
                            float(action["uniform_advantage"])
                            for action in sorted(
                                truth["actions"], key=lambda item: int(item["action"])
                            )
                        )
                        yield ProbeExample(
                            metadata=ProbeMetadata(
                                mdp_index=mdp_index,
                                state_index=state_index,
                                token_index=token_index,
                                source_state_id=prompt.state_id,
                                occupancy=float(truth["root_uniform_occupancy"]),
                                advantage=advantage,
                            ),
                            input_ids=tuple(values),
                        )
                    global_index += 1


def _probe_logits(
    policy,
    examples: Iterator[ProbeExample],
    *,
    batch_size: int,
) -> tuple[list[ProbeMetadata], np.ndarray]:
    import torch

    module = unwrap_policy(policy)
    module.eval()
    metadata: list[ProbeMetadata] = []
    logits: list[np.ndarray] = []
    batch: list[ProbeExample] = []

    def evaluate(items: list[ProbeExample]) -> None:
        input_ids = torch.tensor(
            [item.input_ids for item in items], dtype=torch.long, device="cuda"
        )
        values = module(input_ids).detach().cpu().numpy().astype(np.float32, copy=False)
        metadata.extend(item.metadata for item in items)
        logits.extend(values)

    with torch.inference_mode():
        for example in examples:
            batch.append(example)
            if len(batch) == batch_size:
                evaluate(batch)
                batch = []
        if batch:
            evaluate(batch)
    return metadata, np.asarray(logits, dtype=np.float32)


def _local_score_matrices(
    corpus: ExperimentCorpus,
    metadata: list[ProbeMetadata],
    baseline_bias: np.ndarray,
    updated_logits: np.ndarray,
    *,
    step_size: float,
) -> list[np.ndarray]:
    scores = [
        np.zeros((len(bundle.mdp.states), len(bundle.mdp.states)), dtype=np.float64)
        for bundle in corpus.bundles
    ]
    shifted = updated_logits.astype(np.float32, copy=False) + baseline_bias
    shifted -= shifted.max(axis=1, keepdims=True)
    updated = np.exp(shifted).astype(np.float32)
    updated /= updated.sum(axis=1, keepdims=True)
    uniform = 1.0 / len(corpus.action_token_ids)
    for item, probabilities in zip(metadata, updated, strict=True):
        transfer = item.occupancy * float(
            np.dot(
                probabilities.astype(np.float64) - uniform,
                np.asarray(item.advantage, dtype=np.float64),
            )
        )
        scores[item.mdp_index][item.state_index, item.token_index] += (
            transfer / step_size
        )
    return scores


def _write_geometry_artifact(
    staging: Path,
    corpus: ExperimentCorpus,
    registry: TokenRegistry,
    scores: list[np.ndarray],
    *,
    identity: Mapping[str, Any],
    build_id: str,
    config: ExperimentConfig,
    runtime_reports: list[Mapping[str, Any]],
    reference_loss: float,
    allow_dirty: bool,
    assignment_scope: str,
    workload: Mapping[str, Any],
) -> None:
    tokenizer_path = Path(config.model.path).expanduser().resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    split_by_mdp = {bundle.mdp.mdp_id: bundle.split for bundle in corpus.bundles}
    with (
        JsonlGzipWriter(staging / "labelings.jsonl.gz") as labeling_writer,
        JsonlGzipWriter(staging / "prompts.jsonl.gz") as prompt_writer,
    ):
        score_index: dict[str, str] = {}
        score_payload: dict[str, np.ndarray] = {}
        assignment_summary: dict[str, Any] = {}
        for index, (bundle, matrix) in enumerate(
            zip(corpus.bundles, scores, strict=True)
        ):
            if matrix.shape != (
                len(bundle.mdp.states),
                len(bundle.mdp.states),
            ) or not np.all(np.isfinite(matrix)):
                raise ArtifactError(
                    f"invalid geometry matrix for {bundle.mdp.mdp_id}: {matrix.shape}"
                )
            blocks = _assignment_blocks(
                bundle, registry, assignment_scope=assignment_scope
            )
            key = f"matrix_{index:05d}"
            score_index[bundle.mdp.mdp_id] = key
            score_payload[key] = matrix.astype("<f8", copy=False)
            built_labelings = {}
            for mode in ("aligned", "adversarial"):
                labeling = make_score_aligned_labeling(
                    bundle.mdp,
                    registry,
                    matrix,
                    mode=mode,
                    seed=config.runtime.seed,
                    labeling_id=f"{bundle.mdp.mdp_id}-{mode}-{build_id}",
                    score_provenance={
                        "geometry_build_id": build_id,
                        "method": identity["method"],
                        "model_fingerprint": identity["model_fingerprint"],
                        "probe_step_size": identity["probe_step_size"],
                        "assignment_scope": assignment_scope,
                    },
                    assignment_blocks=blocks,
                )
                built_labelings[mode] = labeling
                labeling_writer.write(
                    {"split": split_by_mdp[bundle.mdp.mdp_id], **labeling.to_dict()}
                )
                prompts = build_all_prompt_records(
                    bundle.mdp,
                    labeling,
                    registry,
                    tokenizer,
                    enable_thinking=False,
                )
                if {len(prompt.input_ids) for prompt in prompts} != {
                    corpus.prompt_length
                }:
                    raise ArtifactError(
                        f"geometry changed prompt length for {bundle.mdp.mdp_id}:{mode}"
                    )
                for prompt in prompts:
                    prompt_writer.write(
                        {"split": split_by_mdp[bundle.mdp.mdp_id], **prompt.to_dict()}
                    )
                if blocks is not None:
                    states = bundle.mdp.ordered_states()
                    for rows, _columns in blocks:
                        base_tokens = {
                            bundle.labeling.state_token_ids[states[row].state_id]
                            for row in rows
                        }
                        assigned_tokens = {
                            labeling.state_tokens[states[row].state_id].token_id
                            for row in rows
                        }
                        if assigned_tokens != base_tokens:
                            raise ArtifactError(
                                "depth-controlled assignment changed a depth token pool "
                                f"for {bundle.mdp.mdp_id}"
                            )
            candidate_column = {
                token.token_id: column
                for column, token in enumerate(
                    registry.state_tokens[: len(bundle.mdp.states)]
                )
            }
            random_columns = [
                candidate_column[bundle.labeling.state_token_ids[state.state_id]]
                for state in bundle.mdp.ordered_states()
            ]
            random_score = float(
                matrix[np.arange(len(random_columns)), random_columns].sum()
            )
            maximum = float(built_labelings["aligned"].assignment_score)
            minimum = float(built_labelings["adversarial"].assignment_score)
            score_range = maximum - minimum
            if not all(
                np.isfinite(value) for value in (maximum, minimum, random_score)
            ):
                raise ArtifactError(
                    f"non-finite assignment score for {bundle.mdp.mdp_id}"
                )
            if score_range <= 0.0:
                raise ArtifactError(
                    f"geometry has no assignment contrast for {bundle.mdp.mdp_id}"
                )
            if not minimum <= random_score <= maximum:
                raise ArtifactError(
                    f"random assignment score lies outside measured endpoints for "
                    f"{bundle.mdp.mdp_id}"
                )
            block_sizes = (
                [len(bundle.mdp.states)]
                if blocks is None
                else [len(rows) for rows, _ in blocks]
            )
            assignment_summary[bundle.mdp.mdp_id] = {
                "assignment_scope": assignment_scope,
                "block_sizes": block_sizes,
                "candidate_pairs": sum(size * size for size in block_sizes),
                "full_candidate_pairs": len(bundle.mdp.states) ** 2,
                "aligned_score": maximum,
                "random_score": random_score,
                "adversarial_score": minimum,
                "score_range": score_range,
                "random_alignment_index": (
                    (random_score - minimum) / score_range if score_range > 0 else None
                ),
            }
    np.savez_compressed(staging / "score_matrices.npz", **score_payload)
    atomic_json(staging / "score_index.json", score_index)
    atomic_json(staging / "assignment_summary.json", assignment_summary)
    provenance = {
        **dict(identity),
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_loss": reference_loss,
        "git": git_provenance(allow_dirty=allow_dirty),
        "runtime_by_rank": runtime_reports,
        "workload": dict(workload),
    }
    atomic_json(staging / "provenance.json", provenance)
    files = []
    for name in (
        "reference-calibration.jsonl.gz",
        "labelings.jsonl.gz",
        "prompts.jsonl.gz",
        "score_matrices.npz",
        "score_index.json",
        "assignment_summary.json",
        "provenance.json",
    ):
        path = staging / name
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_json(
        staging / "manifest.json",
        {
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "build_id": build_id,
            "base_dataset_build_id": corpus.identity.dataset_build_id,
            "token_registry_fingerprint": corpus.identity.token_registry_fingerprint,
            "labeling_modes": ["aligned", "adversarial"],
            "assignment_scope": assignment_scope,
            "instances": len(corpus.bundles),
            "files": files,
        },
    )


def build_geometry(
    config: ExperimentConfig,
    *,
    context: DistributedContext,
    name: str,
    output_root: str | Path,
    allow_dirty: bool,
    assignment_scope: str = "full",
) -> dict[str, Any]:
    import torch

    if config.data.labeling_mode != "random":
        raise RLLabError("geometry must be measured from the paired random labeling")
    if config.model.compute_dtype != "float32":
        raise RLLabError("geometry construction requires model.compute_dtype=float32")
    if assignment_scope not in ASSIGNMENT_SCOPES:
        raise RLLabError(
            f"geometry assignment scope must be one of {sorted(ASSIGNMENT_SCOPES)}"
        )
    corpus = load_experiment_corpus(
        config.data.path,
        split=config.data.split,
        labeling_mode="random",
        max_instances=config.data.max_instances,
    )
    model_fingerprint = model_snapshot_fingerprint(
        config.model.path, config.model.revision
    )
    identity = _geometry_identity(config, corpus, model_fingerprint, assignment_scope)
    build_id = _build_id(identity)
    root = Path(output_root).expanduser().resolve()
    family = root / name
    target = family / build_id
    exists = broadcast_object(target.is_dir() if context.is_main else None, context)
    if exists:
        return {"path": str(target), "build_id": build_id, "reused": True}

    staging = family / f".{build_id}.tmp"
    if context.is_main:
        family.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            raise ArtifactError(f"stale geometry staging directory exists: {staging}")
        staging.mkdir()
    barrier(context)

    registry = TokenRegistry.from_dict(
        json.loads(
            (Path(corpus.identity.dataset_path) / "token_registry.json").read_text(
                encoding="utf-8"
            )
        )
    )
    workload = _geometry_workload(corpus, registry, assignment_scope=assignment_scope)
    if context.is_main:
        print(
            "[geometry] "
            f"scope={assignment_scope} candidate_pairs={workload['candidate_pairs']} "
            f"probe_examples_two_passes={workload['probe_examples_two_passes']}",
            flush=True,
        )
        print("[geometry] loading policy and calibrating random prompts", flush=True)
    policy = load_policy(config.model, corpus.action_token_ids)
    calibration = calibrate_corpus(
        policy,
        corpus,
        context=context,
        model_fingerprint=model_fingerprint,
        batch_size=config.training.forward_batch_states,
        output_path=staging / "reference-calibration.jsonl.gz",
    )

    if context.is_main:
        print("[geometry] measuring baseline depth-block probes", flush=True)
    baseline_metadata, baseline_logits = _probe_logits(
        policy,
        _probe_examples(
            corpus,
            registry,
            rank=context.rank,
            world_size=context.world_size,
            assignment_scope=assignment_scope,
        ),
        batch_size=config.training.forward_batch_states,
    )
    baseline_bias = uniformizing_fixed_bias(baseline_logits)
    if context.is_main:
        print("[geometry] computing exact uniform-policy root gradient", flush=True)
    reference_gradient = exact_uniform_policy_gradient_backward(
        policy,
        corpus,
        calibration,
        context=context,
        microbatch_states=config.training.train_microbatch_states,
    )
    apply_gradient_fraction(policy, step_size=config.training.learning_rate)
    for parameter in unwrap_policy(policy).parameters():
        parameter.grad = None
    torch.cuda.empty_cache()
    if context.is_main:
        print("[geometry] measuring updated depth-block probes", flush=True)
    updated_metadata, updated_logits = _probe_logits(
        policy,
        _probe_examples(
            corpus,
            registry,
            rank=context.rank,
            world_size=context.world_size,
            assignment_scope=assignment_scope,
        ),
        batch_size=config.training.forward_batch_states,
    )
    if updated_metadata != baseline_metadata:
        raise RuntimeError(
            "counterfactual probe order changed across the reference update"
        )
    local_scores = _local_score_matrices(
        corpus,
        baseline_metadata,
        baseline_bias,
        updated_logits,
        step_size=config.training.learning_rate,
    )
    gathered_scores = gather_objects(local_scores, context)
    runtime_reports = gather_objects(runtime_report(context), context)
    if context.is_main:
        print("[geometry] validating and publishing immutable overlay", flush=True)
        scores = [np.zeros_like(matrix) for matrix in local_scores]
        for rank_scores in gathered_scores:
            for index, matrix in enumerate(rank_scores):
                scores[index] += matrix
        _write_geometry_artifact(
            staging,
            corpus,
            registry,
            scores,
            identity=identity,
            build_id=build_id,
            config=config,
            runtime_reports=runtime_reports,
            reference_loss=reference_gradient.loss,
            allow_dirty=allow_dirty,
            assignment_scope=assignment_scope,
            workload=workload,
        )
        with storage_lock(root, f"geometry-{name}"):
            os.replace(staging, target)
            link = family / "latest"
            temporary_link = family / ".latest.tmp"
            if temporary_link.is_symlink():
                temporary_link.unlink()
            temporary_link.symlink_to(build_id, target_is_directory=True)
            os.replace(temporary_link, link)
    barrier(context)
    return {
        "path": str(target),
        "build_id": build_id,
        "reused": False,
        "assignment_scope": assignment_scope,
        "workload": workload,
    }
