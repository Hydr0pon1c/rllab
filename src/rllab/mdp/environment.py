"""Pure-Python environment with exact reset, snapshot, and counterfactual APIs."""

from __future__ import annotations

from dataclasses import dataclass

from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP, Transition, validate_mdp


@dataclass(frozen=True)
class EnvSnapshot:
    mdp_id: str
    state_id: str
    step_count: int
    transition_cost: int
    terminal_cost_paid: int


@dataclass(frozen=True)
class StepResult:
    source_state_id: str
    action: int
    next_state_id: str
    immediate_cost: int
    terminal_cost: int
    incurred_cost: int
    reward: float
    done: bool


class PuzzleEnv:
    """A minimal deterministic runtime whose state can be copied exactly."""

    def __init__(self, mdp: PuzzleMDP):
        validate_mdp(mdp)
        self.mdp = mdp
        self._states = {state.state_id: state for state in mdp.states}
        self._outgoing = mdp.outgoing()
        self._state_id = mdp.start_state
        self._step_count = 0
        self._transition_cost = 0
        self._terminal_cost_paid = 0

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def done(self) -> bool:
        return self._states[self._state_id].terminal

    @property
    def total_cost(self) -> int:
        return self._transition_cost + self._terminal_cost_paid

    def reset(self, state_id: str | None = None) -> str:
        resolved = state_id or self.mdp.start_state
        if resolved not in self._states:
            raise ValidationError(f"cannot reset to unknown state {resolved!r}")
        self._state_id = resolved
        self._step_count = 0
        self._transition_cost = 0
        self._terminal_cost_paid = 0
        return self._state_id

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            mdp_id=self.mdp.mdp_id,
            state_id=self._state_id,
            step_count=self._step_count,
            transition_cost=self._transition_cost,
            terminal_cost_paid=self._terminal_cost_paid,
        )

    def restore(self, snapshot: EnvSnapshot) -> str:
        if snapshot.mdp_id != self.mdp.mdp_id:
            raise ValidationError(
                f"snapshot belongs to {snapshot.mdp_id!r}, not {self.mdp.mdp_id!r}"
            )
        if snapshot.state_id not in self._states:
            raise ValidationError(
                f"snapshot references unknown state {snapshot.state_id!r}"
            )
        if (
            snapshot.step_count < 0
            or snapshot.transition_cost < 0
            or snapshot.terminal_cost_paid < 0
        ):
            raise ValidationError("snapshot counters cannot be negative")
        self._state_id = snapshot.state_id
        self._step_count = snapshot.step_count
        self._transition_cost = snapshot.transition_cost
        self._terminal_cost_paid = snapshot.terminal_cost_paid
        return self._state_id

    def _preview_transition(self, transition: Transition) -> StepResult:
        target = self._states[transition.target]
        terminal_cost = target.terminal_cost if target.terminal else 0
        incurred = transition.cost + terminal_cost
        return StepResult(
            source_state_id=transition.source,
            action=transition.action,
            next_state_id=transition.target,
            immediate_cost=transition.cost,
            terminal_cost=terminal_cost,
            incurred_cost=incurred,
            reward=-float(incurred),
            done=target.terminal,
        )

    def counterfactuals(self) -> tuple[StepResult, ...]:
        """Return every possible next-step result without mutating the env."""

        if self.done:
            return ()
        return tuple(
            self._preview_transition(transition)
            for transition in self._outgoing[self._state_id]
        )

    def step(self, action: int) -> StepResult:
        if self.done:
            raise ValidationError(
                "cannot step a terminal environment; call reset/restore"
            )
        if not 0 <= action < self.mdp.num_actions:
            raise ValidationError(
                f"action {action} is outside [0, {self.mdp.num_actions})"
            )
        transition = self._outgoing[self._state_id][action]
        result = self._preview_transition(transition)
        self._state_id = result.next_state_id
        self._step_count += 1
        self._transition_cost += result.immediate_cost
        self._terminal_cost_paid += result.terminal_cost
        return result
