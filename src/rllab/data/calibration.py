"""Reference masked-softmax and frozen initial-policy calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from rllab.errors import ValidationError

CALIBRATION_SCHEMA_VERSION = "initial-action-calibration/v1"


@dataclass(frozen=True)
class ActionCalibrationRecord:
    """One auditable frozen-checkpoint calibration row."""

    model_fingerprint: str
    token_registry_fingerprint: str
    mdp_id: str
    labeling_id: str
    state_id: str
    prompt_sha256: str
    action_token_ids: tuple[int, ...]
    initial_logits_fp32: tuple[float, ...]
    fixed_bias_fp32: tuple[float, ...]
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_token_ids"] = list(self.action_token_ids)
        result["initial_logits_fp32"] = list(self.initial_logits_fp32)
        result["fixed_bias_fp32"] = list(self.fixed_bias_fp32)
        return result


def masked_softmax(
    logits: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    *,
    fixed_bias: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """Numerically stable FP32 reference shared by rollout/loss tests."""

    values = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if values.shape != mask.shape:
        raise ValidationError(f"logits shape {values.shape} != mask shape {mask.shape}")
    if fixed_bias is not None:
        bias = np.asarray(fixed_bias, dtype=np.float32)
        if bias.shape != values.shape:
            raise ValidationError(
                f"bias shape {bias.shape} != logits shape {values.shape}"
            )
        values = values + bias
    if np.any(mask.sum(axis=-1) == 0):
        raise ValidationError(
            "every masked-softmax row needs at least one valid action"
        )
    masked = np.where(mask, values, -np.inf)
    maxima = np.max(masked, axis=-1, keepdims=True)
    exponentials = np.where(mask, np.exp(masked - maxima), 0.0).astype(np.float32)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def uniformizing_fixed_bias(
    initial_action_logits: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Return immutable per-state/action biases yielding an exact uniform policy.

    The row mean is retained only to keep the correction centred; softmax is
    invariant to this shared constant. Call this once on the frozen initial
    checkpoint and persist the result with model/prompt fingerprints.
    """

    logits = np.asarray(initial_action_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValidationError(
            "initial action logits must have shape [states, actions>=2]"
        )
    if not np.all(np.isfinite(logits)):
        raise ValidationError("initial action logits contain non-finite values")
    return logits.mean(axis=1, keepdims=True) - logits


def build_calibration_records(
    *,
    model_fingerprint: str,
    token_registry_fingerprint: str,
    mdp_id: str,
    labeling_id: str,
    state_ids: Sequence[str],
    prompt_sha256: Sequence[str],
    action_token_ids: Sequence[int],
    initial_action_logits: NDArray[np.floating],
) -> tuple[ActionCalibrationRecord, ...]:
    """Convert a GPU probe into portable FP32 rows and verify uniformisation."""

    logits = np.asarray(initial_action_logits, dtype=np.float32)
    expected_shape = (len(state_ids), len(action_token_ids))
    if logits.shape != expected_shape:
        raise ValidationError(
            f"initial logits shape {logits.shape}, expected {expected_shape}"
        )
    if len(prompt_sha256) != len(state_ids):
        raise ValidationError("prompt hashes and state IDs have different lengths")
    if len(set(state_ids)) != len(state_ids):
        raise ValidationError("calibration state IDs must be unique")
    if len(set(action_token_ids)) != len(action_token_ids):
        raise ValidationError("calibration action token IDs must be unique")
    bias = uniformizing_fixed_bias(logits)
    calibrated = masked_softmax(
        logits,
        np.ones_like(logits, dtype=np.bool_),
        fixed_bias=bias,
    )
    expected = np.full_like(calibrated, 1.0 / len(action_token_ids))
    if not np.allclose(calibrated, expected, atol=2e-7, rtol=0.0):
        raise ValidationError(
            "computed fixed bias does not produce a uniform FP32 policy"
        )
    return tuple(
        ActionCalibrationRecord(
            model_fingerprint=model_fingerprint,
            token_registry_fingerprint=token_registry_fingerprint,
            mdp_id=mdp_id,
            labeling_id=labeling_id,
            state_id=state_id,
            prompt_sha256=prompt_hash,
            action_token_ids=tuple(int(value) for value in action_token_ids),
            initial_logits_fp32=tuple(float(value) for value in logits[index]),
            fixed_bias_fp32=tuple(float(value) for value in bias[index]),
        )
        for index, (state_id, prompt_hash) in enumerate(
            zip(state_ids, prompt_sha256, strict=True)
        )
    )
