"""Canonical, tokenizer-independent PuzzleMDP schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from rllab.errors import ValidationError

MDP_SCHEMA_VERSION = "puzzle-mdp/v1"


@dataclass(frozen=True, order=True)
class State:
    state_id: str
    depth: int
    terminal: bool = False
    terminal_cost: int = 0


@dataclass(frozen=True, order=True)
class Transition:
    source: str
    action: int
    target: str
    cost: int


@dataclass(frozen=True)
class PuzzleMDP:
    mdp_id: str
    seed: int
    start_state: str
    horizon: int
    num_actions: int
    states: tuple[State, ...]
    transitions: tuple[Transition, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MDP_SCHEMA_VERSION

    def ordered_states(self) -> tuple[State, ...]:
        return tuple(
            sorted(self.states, key=lambda state: (state.depth, state.state_id))
        )

    def outgoing(self) -> dict[str, tuple[Transition, ...]]:
        result: dict[str, list[Transition]] = {
            state.state_id: [] for state in self.states
        }
        for transition in self.transitions:
            result.setdefault(transition.source, []).append(transition)
        return {
            state_id: tuple(sorted(items, key=lambda transition: transition.action))
            for state_id, items in result.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mdp_id": self.mdp_id,
            "seed": self.seed,
            "start_state": self.start_state,
            "horizon": self.horizon,
            "num_actions": self.num_actions,
            "states": [asdict(state) for state in self.ordered_states()],
            "transitions": [
                asdict(transition)
                for transition in sorted(
                    self.transitions,
                    key=lambda item: (item.source, item.action, item.target),
                )
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PuzzleMDP":
        schema_version = raw.get("schema_version")
        if schema_version != MDP_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported MDP schema {schema_version!r}; expected {MDP_SCHEMA_VERSION!r}"
            )
        try:
            mdp = cls(
                schema_version=schema_version,
                mdp_id=str(raw["mdp_id"]),
                seed=int(raw["seed"]),
                start_state=str(raw["start_state"]),
                horizon=int(raw["horizon"]),
                num_actions=int(raw["num_actions"]),
                states=tuple(State(**record) for record in raw["states"]),
                transitions=tuple(
                    Transition(**record) for record in raw["transitions"]
                ),
                metadata=dict(raw.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"malformed PuzzleMDP record: {exc}") from exc
        validate_mdp(mdp)
        return mdp


def validate_mdp(mdp: PuzzleMDP) -> None:
    """Validate the exact layered-DAG invariants used by schema v1."""

    if mdp.schema_version != MDP_SCHEMA_VERSION:
        raise ValidationError(f"unexpected schema version: {mdp.schema_version}")
    if mdp.horizon < 1 or mdp.num_actions < 2:
        raise ValidationError("horizon must be >= 1 and num_actions must be >= 2")

    state_by_id = {state.state_id: state for state in mdp.states}
    if len(state_by_id) != len(mdp.states):
        raise ValidationError("state IDs are not unique")
    if mdp.start_state not in state_by_id:
        raise ValidationError(f"start state {mdp.start_state!r} does not exist")
    if state_by_id[mdp.start_state].depth != 0:
        raise ValidationError("start state must be at depth 0")

    outgoing: dict[str, list[Transition]] = {state_id: [] for state_id in state_by_id}
    for transition in mdp.transitions:
        if transition.source not in state_by_id or transition.target not in state_by_id:
            raise ValidationError(
                f"transition references an unknown state: {transition}"
            )
        source = state_by_id[transition.source]
        target = state_by_id[transition.target]
        if source.terminal:
            raise ValidationError(
                f"terminal state {source.state_id} has outgoing transitions"
            )
        if target.depth != source.depth + 1:
            raise ValidationError(
                f"transition is not between adjacent layers: {transition}"
            )
        if not 0 <= transition.action < mdp.num_actions:
            raise ValidationError(f"action index out of range: {transition}")
        if transition.cost < 0:
            raise ValidationError(f"negative transition cost: {transition}")
        outgoing[transition.source].append(transition)

    for state in mdp.states:
        if not 0 <= state.depth <= mdp.horizon:
            raise ValidationError(
                f"state {state.state_id} has invalid depth {state.depth}"
            )
        expected_terminal = state.depth == mdp.horizon
        if state.terminal != expected_terminal:
            raise ValidationError(
                f"state {state.state_id} terminal={state.terminal}, expected {expected_terminal}"
            )
        if state.terminal_cost < 0:
            raise ValidationError(
                f"state {state.state_id} has a negative terminal cost"
            )
        if not state.terminal and state.terminal_cost != 0:
            raise ValidationError(
                f"non-terminal state {state.state_id} has a terminal cost"
            )
        actions = [transition.action for transition in outgoing[state.state_id]]
        expected_actions = [] if state.terminal else list(range(mdp.num_actions))
        if sorted(actions) != expected_actions:
            raise ValidationError(
                f"state {state.state_id} actions are {sorted(actions)}, expected {expected_actions}"
            )
        targets = [transition.target for transition in outgoing[state.state_id]]
        if len(targets) != len(set(targets)):
            raise ValidationError(
                f"state {state.state_id} has duplicate action destinations"
            )

    reachable = {mdp.start_state}
    for state in mdp.ordered_states():
        if state.state_id not in reachable:
            continue
        reachable.update(transition.target for transition in outgoing[state.state_id])
    missing = set(state_by_id) - reachable
    if missing:
        preview = sorted(missing)[:8]
        raise ValidationError(
            f"{len(missing)} states are unreachable from start: {preview}"
        )


def state_ids_at_depth(states: Iterable[State], depth: int) -> tuple[str, ...]:
    return tuple(sorted(state.state_id for state in states if state.depth == depth))
