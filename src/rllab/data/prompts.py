"""Build and audit history-free, masked-action prompt records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from rllab.data.relabel import StateLabeling
from rllab.data.template import (
    ACTIONS_HEADER,
    COST_FRAGMENT,
    CURRENT_STATE_PREFIX,
    GO_TO_FRAGMENT,
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
)
from rllab.data.tokens import (
    TokenRegistry,
    encode_chat,
    encode_with_offsets,
    render_chat,
)
from rllab.errors import ValidationError
from rllab.mdp.schema import PuzzleMDP

PROMPT_RECORD_SCHEMA_VERSION = "prompt-record/v1"


@dataclass(frozen=True)
class SlotSpan:
    role: str
    canonical_id: str
    char_start: int
    char_end: int
    token_position: int | None = None
    action_index: int | None = None


@dataclass(frozen=True)
class PromptRecord:
    example_id: str
    mdp_id: str
    labeling_id: str
    state_id: str
    depth: int
    messages: tuple[Mapping[str, str], ...]
    rendered_prompt: str
    input_ids: tuple[int, ...]
    slots: tuple[SlotSpan, ...]
    allowed_action_token_ids: tuple[int, ...]
    prompt_sha256: str
    schema_version: str = PROMPT_RECORD_SCHEMA_VERSION
    template_version: str = PROMPT_TEMPLATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_version": self.template_version,
            "example_id": self.example_id,
            "mdp_id": self.mdp_id,
            "labeling_id": self.labeling_id,
            "state_id": self.state_id,
            "depth": self.depth,
            "messages": [dict(message) for message in self.messages],
            "rendered_prompt": self.rendered_prompt,
            "input_ids": list(self.input_ids),
            "slots": [asdict(slot) for slot in self.slots],
            "allowed_action_token_ids": list(self.allowed_action_token_ids),
            "prompt_sha256": self.prompt_sha256,
        }


def _append_slot(
    chunks: list[str],
    spans: list[SlotSpan],
    *,
    text: str,
    role: str,
    canonical_id: str,
    action_index: int | None = None,
) -> None:
    start = sum(len(chunk) for chunk in chunks)
    chunks.append(text)
    spans.append(
        SlotSpan(
            role=role,
            canonical_id=canonical_id,
            char_start=start,
            char_end=start + len(text),
            action_index=action_index,
        )
    )


def build_user_content(
    mdp: PuzzleMDP,
    labeling: StateLabeling,
    state_id: str,
) -> tuple[str, tuple[SlotSpan, ...]]:
    state_by_id = {state.state_id: state for state in mdp.states}
    if state_id not in state_by_id:
        raise ValidationError(f"cannot build a prompt for unknown state {state_id!r}")
    if state_by_id[state_id].terminal:
        raise ValidationError(f"terminal state {state_id!r} has no action prompt")
    if labeling.mdp_id != mdp.mdp_id:
        raise ValidationError("labeling and MDP IDs do not match")

    chunks = [CURRENT_STATE_PREFIX]
    spans: list[SlotSpan] = []
    _append_slot(
        chunks,
        spans,
        text=labeling.state_tokens[state_id].text,
        role="current_state",
        canonical_id=state_id,
    )
    chunks.extend(["\n", ACTIONS_HEADER])
    outgoing = mdp.outgoing()[state_id]
    for line_index, transition in enumerate(outgoing):
        if line_index:
            chunks.append("\n")
        action_token = labeling.action_tokens[transition.action]
        _append_slot(
            chunks,
            spans,
            text=action_token.text,
            role="action",
            canonical_id=str(transition.action),
            action_index=transition.action,
        )
        chunks.append(GO_TO_FRAGMENT)
        _append_slot(
            chunks,
            spans,
            text=labeling.state_tokens[transition.target].text,
            role="destination_state",
            canonical_id=transition.target,
            action_index=transition.action,
        )
        chunks.extend([COST_FRAGMENT, str(transition.cost)])
    return "".join(chunks), tuple(spans)


def build_prompt_record(
    mdp: PuzzleMDP,
    labeling: StateLabeling,
    registry: TokenRegistry,
    tokenizer,
    state_id: str,
    *,
    enable_thinking: bool,
) -> PromptRecord:
    if labeling.registry_fingerprint != registry.fingerprint:
        raise ValidationError("labeling was created from a different token registry")
    user_content, local_spans = build_user_content(mdp, labeling, state_id)
    messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )
    rendered = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    first = rendered.find(user_content)
    if first < 0 or rendered.find(user_content, first + 1) >= 0:
        raise ValidationError(
            "could not uniquely locate user content in rendered chat prompt"
        )
    input_ids, offsets = encode_with_offsets(tokenizer, rendered)
    templated_ids = encode_chat(tokenizer, messages, enable_thinking=enable_thinking)
    if input_ids != templated_ids:
        raise ValidationError(
            "raw rendered tokenization differs from apply_chat_template tokenization"
        )

    audited_spans: list[SlotSpan] = []
    placeholder_positions: set[int] = set()
    state_token_ids = {token.token_id for token in labeling.state_tokens.values()}
    action_token_ids = {token.token_id for token in labeling.action_tokens}
    for span in local_spans:
        absolute_start = first + span.char_start
        absolute_end = first + span.char_end
        positions = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > absolute_start and left < absolute_end
        ]
        if len(positions) != 1 or offsets[positions[0]] != (
            absolute_start,
            absolute_end,
        ):
            raise ValidationError(
                f"placeholder {span.role}:{span.canonical_id} is not exactly one token"
            )
        position = positions[0]
        expected_token = (
            labeling.action_tokens[int(span.canonical_id)]
            if span.role == "action"
            else labeling.state_tokens[span.canonical_id]
        )
        if input_ids[position] != expected_token.token_id:
            raise ValidationError(
                f"placeholder {span.role}:{span.canonical_id} encoded as {input_ids[position]}, "
                f"expected {expected_token.token_id}"
            )
        placeholder_positions.add(position)
        audited_spans.append(
            SlotSpan(
                role=span.role,
                canonical_id=span.canonical_id,
                char_start=absolute_start,
                char_end=absolute_end,
                token_position=position,
                action_index=span.action_index,
            )
        )

    static_ids = {
        token_id
        for position, token_id in enumerate(input_ids)
        if position not in placeholder_positions
    }
    overlap = static_ids & (state_token_ids | action_token_ids)
    if overlap:
        raise ValidationError(
            f"reserved state/action IDs appear in static prompt tokens: {sorted(overlap)}"
        )
    if state_token_ids & action_token_ids:
        raise ValidationError("state and action token IDs overlap")

    state = next(item for item in mdp.states if item.state_id == state_id)
    prompt_sha = sha256(
        json.dumps(input_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return PromptRecord(
        example_id=f"{labeling.labeling_id}:{state_id}",
        mdp_id=mdp.mdp_id,
        labeling_id=labeling.labeling_id,
        state_id=state_id,
        depth=state.depth,
        messages=messages,
        rendered_prompt=rendered,
        input_ids=tuple(input_ids),
        slots=tuple(audited_spans),
        allowed_action_token_ids=tuple(
            token.token_id for token in labeling.action_tokens
        ),
        prompt_sha256=prompt_sha,
    )


def build_all_prompt_records(
    mdp: PuzzleMDP,
    labeling: StateLabeling,
    registry: TokenRegistry,
    tokenizer,
    *,
    enable_thinking: bool,
) -> tuple[PromptRecord, ...]:
    records = tuple(
        build_prompt_record(
            mdp,
            labeling,
            registry,
            tokenizer,
            state.state_id,
            enable_thinking=enable_thinking,
        )
        for state in mdp.ordered_states()
        if not state.terminal
    )
    lengths = {len(record.input_ids) for record in records}
    if len(lengths) != 1:
        raise ValidationError(
            f"tokenized prompts do not have invariant length: {sorted(lengths)}"
        )
    return records
