"""Exact current/destination/absent exposure accounting for update batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP

EXPOSURE_SCHEMA_VERSION = "state-exposure/v1"


class ExposureClass(str, Enum):
    POLICY_LOSS_CURRENT = "policy_loss_current"
    DESTINATION_CONTEXT_ONLY = "destination_context_only"
    TOKEN_ABSENT = "token_absent"


@dataclass(frozen=True)
class StateExposure:
    state_id: str
    exposure_class: ExposureClass
    current_count: int
    destination_count: int
    schema_version: str = EXPOSURE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["exposure_class"] = self.exposure_class.value
        return result


def classify_state_exposure(
    mdp: PuzzleMDP,
    policy_loss_current_states: Iterable[str],
) -> tuple[StateExposure, ...]:
    """Partition all states into the three non-overlapping research classes."""

    state_ids = {state.state_id for state in mdp.states}
    current_counts = {state_id: 0 for state_id in state_ids}
    for state_id in policy_loss_current_states:
        if state_id not in state_ids:
            raise ValidationError(f"batch contains unknown current state {state_id!r}")
        state = next(item for item in mdp.states if item.state_id == state_id)
        if state.terminal:
            raise ValidationError(
                f"terminal state {state_id!r} cannot receive a policy loss"
            )
        current_counts[state_id] += 1

    destination_counts = {state_id: 0 for state_id in state_ids}
    outgoing = mdp.outgoing()
    for state_id, count in current_counts.items():
        if count == 0:
            continue
        for transition in outgoing[state_id]:
            destination_counts[transition.target] += count

    records: list[StateExposure] = []
    for state in mdp.ordered_states():
        current_count = current_counts[state.state_id]
        destination_count = destination_counts[state.state_id]
        if current_count:
            exposure_class = ExposureClass.POLICY_LOSS_CURRENT
        elif destination_count:
            exposure_class = ExposureClass.DESTINATION_CONTEXT_ONLY
        else:
            exposure_class = ExposureClass.TOKEN_ABSENT
        records.append(
            StateExposure(
                state_id=state.state_id,
                exposure_class=exposure_class,
                current_count=current_count,
                destination_count=destination_count,
            )
        )
    return tuple(records)
