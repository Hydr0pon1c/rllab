"""Distributed initial-policy calibration with immutable prompt bindings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rllab.artifacts import JsonlGzipWriter, iter_jsonl_gz
from rllab.data.calibration import uniformizing_fixed_bias
from rllab.errors import ArtifactError, ValidationError
from rllab.experiment.corpus import ExperimentCorpus, PromptExample, prompts_for_rank
from rllab.experiment.modeling import unwrap_policy
from rllab.experiment.runtime import DistributedContext, barrier, gather_objects


@dataclass(frozen=True)
class CalibrationRow:
    example_id: str
    mdp_id: str
    labeling_id: str
    state_id: str
    prompt_sha256: str
    action_token_ids: tuple[int, ...]
    initial_logits: tuple[float, ...]
    fixed_bias: tuple[float, ...]

    def to_dict(
        self, *, model_fingerprint: str, token_registry_fingerprint: str
    ) -> dict[str, Any]:
        return {
            "schema_version": "initial-action-calibration/v1",
            "model_fingerprint": model_fingerprint,
            "token_registry_fingerprint": token_registry_fingerprint,
            "example_id": self.example_id,
            "mdp_id": self.mdp_id,
            "labeling_id": self.labeling_id,
            "state_id": self.state_id,
            "prompt_sha256": self.prompt_sha256,
            "action_token_ids": list(self.action_token_ids),
            "initial_logits_fp32": list(self.initial_logits),
            "fixed_bias_fp32": list(self.fixed_bias),
        }


@dataclass(frozen=True)
class CalibrationTable:
    rows: Mapping[str, CalibrationRow]
    model_fingerprint: str
    token_registry_fingerprint: str

    def bias(self, example_id: str) -> np.ndarray:
        try:
            return np.asarray(self.rows[example_id].fixed_bias, dtype=np.float32)
        except KeyError as exc:
            raise ValidationError(f"calibration is missing {example_id}") from exc


def _batch_logits(
    policy, prompts: tuple[PromptExample, ...], batch_size: int
) -> list[list[float]]:
    import torch

    module = unwrap_policy(policy)
    module.eval()
    results: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            input_ids = torch.tensor(
                [prompt.input_ids for prompt in batch], dtype=torch.long, device="cuda"
            )
            logits = (
                module(input_ids).detach().cpu().numpy().astype(np.float32, copy=False)
            )
            results.extend(logits.tolist())
    return results


def calibrate_corpus(
    policy,
    corpus: ExperimentCorpus,
    *,
    context: DistributedContext,
    model_fingerprint: str,
    batch_size: int,
    output_path: str | Path,
) -> CalibrationTable:
    all_prompts = corpus.all_prompts()
    local_prompts = prompts_for_rank(
        all_prompts, rank=context.rank, world_size=context.world_size
    )
    local_logits = _batch_logits(policy, local_prompts, batch_size)
    local_rows: list[dict[str, Any]] = []
    if local_logits:
        logits_array = np.asarray(local_logits, dtype=np.float32)
        biases = uniformizing_fixed_bias(logits_array)
        for prompt, logits, bias in zip(
            local_prompts, logits_array, biases, strict=True
        ):
            local_rows.append(
                CalibrationRow(
                    example_id=prompt.example_id,
                    mdp_id=prompt.mdp_id,
                    labeling_id=prompt.labeling_id,
                    state_id=prompt.state_id,
                    prompt_sha256=prompt.prompt_sha256,
                    action_token_ids=prompt.action_token_ids,
                    initial_logits=tuple(float(value) for value in logits),
                    fixed_bias=tuple(float(value) for value in bias),
                ).to_dict(
                    model_fingerprint=model_fingerprint,
                    token_registry_fingerprint=corpus.identity.token_registry_fingerprint,
                )
            )
    gathered = gather_objects(local_rows, context)
    if context.is_main:
        rows = [row for rank_rows in gathered for row in rank_rows]
        rows.sort(key=lambda row: row["example_id"])
        if len(rows) != len(all_prompts) or len(
            {row["example_id"] for row in rows}
        ) != len(rows):
            raise ArtifactError(
                "distributed calibration did not produce one row per prompt"
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with JsonlGzipWriter(output) as writer:
            for row in rows:
                writer.write(row)
    barrier(context)
    return load_calibration(
        output_path,
        corpus=corpus,
        model_fingerprint=model_fingerprint,
    )


def load_calibration(
    path: str | Path,
    *,
    corpus: ExperimentCorpus,
    model_fingerprint: str,
) -> CalibrationTable:
    rows: dict[str, CalibrationRow] = {}
    prompt_by_id = {prompt.example_id: prompt for prompt in corpus.all_prompts()}
    for raw in iter_jsonl_gz(path):
        example_id = str(raw["example_id"])
        if raw.get("model_fingerprint") != model_fingerprint:
            raise ArtifactError("calibration model fingerprint mismatch")
        if (
            raw.get("token_registry_fingerprint")
            != corpus.identity.token_registry_fingerprint
        ):
            raise ArtifactError("calibration token registry fingerprint mismatch")
        if example_id not in prompt_by_id:
            raise ArtifactError(f"calibration contains unknown prompt {example_id}")
        prompt = prompt_by_id[example_id]
        if raw.get("prompt_sha256") != prompt.prompt_sha256:
            raise ArtifactError(f"calibration prompt hash mismatch for {example_id}")
        if tuple(raw["action_token_ids"]) != prompt.action_token_ids:
            raise ArtifactError(f"calibration action IDs mismatch for {example_id}")
        row = CalibrationRow(
            example_id=example_id,
            mdp_id=str(raw["mdp_id"]),
            labeling_id=str(raw["labeling_id"]),
            state_id=str(raw["state_id"]),
            prompt_sha256=str(raw["prompt_sha256"]),
            action_token_ids=tuple(int(value) for value in raw["action_token_ids"]),
            initial_logits=tuple(float(value) for value in raw["initial_logits_fp32"]),
            fixed_bias=tuple(float(value) for value in raw["fixed_bias_fp32"]),
        )
        if example_id in rows:
            raise ArtifactError(f"duplicate calibration row {example_id}")
        calibrated = np.asarray(row.initial_logits, dtype=np.float32) + np.asarray(
            row.fixed_bias, dtype=np.float32
        )
        probabilities = np.exp(calibrated - calibrated.max())
        probabilities /= probabilities.sum()
        expected = np.full_like(probabilities, 1.0 / len(probabilities))
        if not np.allclose(probabilities, expected, atol=2e-7, rtol=0.0):
            raise ArtifactError(f"calibration is not uniform for {example_id}")
        rows[example_id] = row
    if set(rows) != set(prompt_by_id):
        raise ArtifactError("calibration coverage differs from selected corpus")
    return CalibrationTable(
        rows=rows,
        model_fingerprint=model_fingerprint,
        token_registry_fingerprint=corpus.identity.token_registry_fingerprint,
    )
