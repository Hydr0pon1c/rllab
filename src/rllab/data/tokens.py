"""Qwen token registry creation and boundary-invariance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import unicodedata
from typing import Any, Mapping, Sequence

from rllab.data.template import (
    COST_FRAGMENT,
    CURRENT_STATE_PREFIX,
    GO_TO_FRAGMENT,
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    fixed_text_fragments,
)
from rllab.errors import ValidationError

TOKEN_REGISTRY_SCHEMA_VERSION = "token-registry/v1"


@dataclass(frozen=True, order=True)
class TokenSpec:
    text: str
    token_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenRegistry:
    source: str
    revision: str
    tokenizer_path: str
    tokenizer_class: str
    tokenizer_fingerprint: str
    prompt_template_version: str
    action_tokens: tuple[TokenSpec, ...]
    state_tokens: tuple[TokenSpec, ...]
    fixed_token_ids: tuple[int, ...]
    special_token_ids: tuple[int, ...]
    pool_seed: int
    schema_version: str = TOKEN_REGISTRY_SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        payload = self.to_dict(include_fingerprint=False)
        # The same pinned tokenizer bytes are the same registry even when the
        # persistent disk is mounted at another absolute path.
        payload.pop("tokenizer_path", None)
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "source": self.source,
            "revision": self.revision,
            "tokenizer_path": self.tokenizer_path,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "prompt_template_version": self.prompt_template_version,
            "action_tokens": [token.to_dict() for token in self.action_tokens],
            "state_tokens": [token.to_dict() for token in self.state_tokens],
            "fixed_token_ids": list(self.fixed_token_ids),
            "special_token_ids": list(self.special_token_ids),
            "pool_seed": self.pool_seed,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TokenRegistry":
        if raw.get("schema_version") != TOKEN_REGISTRY_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported token registry schema: {raw.get('schema_version')}"
            )
        registry = cls(
            source=str(raw["source"]),
            revision=str(raw["revision"]),
            tokenizer_path=str(raw["tokenizer_path"]),
            tokenizer_class=str(raw["tokenizer_class"]),
            tokenizer_fingerprint=str(raw["tokenizer_fingerprint"]),
            prompt_template_version=str(raw["prompt_template_version"]),
            action_tokens=tuple(TokenSpec(**item) for item in raw["action_tokens"]),
            state_tokens=tuple(TokenSpec(**item) for item in raw["state_tokens"]),
            fixed_token_ids=tuple(int(value) for value in raw["fixed_token_ids"]),
            special_token_ids=tuple(int(value) for value in raw["special_token_ids"]),
            pool_seed=int(raw["pool_seed"]),
        )
        expected = raw.get("fingerprint")
        if expected is not None and registry.fingerprint != expected:
            raise ValidationError(
                "token registry fingerprint does not match its content"
            )
        validate_token_registry(registry)
        return registry


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_files_fingerprint(path: str | Path) -> str:
    root = Path(path).resolve()
    names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    records: list[tuple[str, str]] = []
    for name in names:
        candidate = root / name
        if candidate.is_file():
            records.append((name, _sha256_file(candidate)))
    if not records:
        raise ValidationError(f"no tokenizer files found under {root}")
    payload = json.dumps(records, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def load_tokenizer(path: str | Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ValidationError(
            "transformers is required for token auditing; install rllab-puzzle[tokenizer]"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValidationError(
            "a fast tokenizer is required for exact placeholder offset auditing"
        )
    if not getattr(tokenizer, "chat_template", None):
        raise ValidationError("the tokenizer has no chat template")
    return tokenizer


def _input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, list):
        return [int(value) for value in encoded]
    if isinstance(encoded, Mapping) or hasattr(encoded, "__getitem__"):
        return [int(value) for value in encoded["input_ids"]]
    raise ValidationError(f"unexpected tokenizer output type: {type(encoded).__name__}")


def encode_chat(
    tokenizer, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    return _input_ids(encoded)


def render_chat(
    tokenizer, messages: Sequence[Mapping[str, str]], *, enable_thinking: bool
) -> str:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if not isinstance(rendered, str):
        raise ValidationError("chat template did not render a string")
    return rendered


def encode_with_offsets(
    tokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    ids = _input_ids(encoded)
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return ids, offsets


def _single_token_in_context(
    tokenizer, token: TokenSpec, prefix: str, suffix: str
) -> bool:
    text = prefix + token.text + suffix
    ids, offsets = encode_with_offsets(tokenizer, text)
    start = len(prefix)
    end = start + len(token.text)
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]
    return (
        len(positions) == 1
        and ids[positions[0]] == token.token_id
        and offsets[positions[0]] == (start, end)
    )


def _state_candidate_text(text: str) -> bool:
    if len(text) != 1 or text.isascii() or not text.isprintable() or text.isspace():
        return False
    # Keep the pool in one Unicode script so random relabeling does not mix
    # visually/linguistically different token classes. Qwen3-8B exposes >8k
    # boundary-stable one-token Han characters, enough for the intended tasks.
    return unicodedata.name(text, "").startswith("CJK UNIFIED IDEOGRAPH")


def _hash_rank(seed: int, fingerprint: str, token_id: int) -> bytes:
    return sha256(f"{seed}:{fingerprint}:{token_id}".encode("ascii")).digest()


def _static_token_ids(tokenizer, *, enable_thinking: bool) -> set[int]:
    result: set[int] = set()
    for fragment in fixed_text_fragments():
        result.update(tokenizer.encode(fragment, add_special_tokens=False))
    scaffold_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ""},
    ]
    result.update(
        encode_chat(tokenizer, scaffold_messages, enable_thinking=enable_thinking)
    )
    return {int(value) for value in result}


def build_token_registry(
    tokenizer,
    *,
    tokenizer_path: str | Path,
    source: str,
    revision: str,
    action_labels: Sequence[str],
    state_pool_size: int,
    pool_seed: int,
    enable_thinking: bool,
) -> TokenRegistry:
    """Select a deterministic pool and prove each label remains one token."""

    root = Path(tokenizer_path).resolve()
    tokenizer_fingerprint = tokenizer_files_fingerprint(root)
    fixed_ids = _static_token_ids(tokenizer, enable_thinking=enable_thinking)
    special_ids = {int(value) for value in tokenizer.all_special_ids}

    action_tokens: list[TokenSpec] = []
    for label in action_labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValidationError(f"action label {label!r} is not one token: {ids}")
        token = TokenSpec(text=label, token_id=int(ids[0]))
        contexts = (("\n", GO_TO_FRAGMENT), ("", GO_TO_FRAGMENT), ("\n\n", ""))
        if not all(
            _single_token_in_context(tokenizer, token, prefix, suffix)
            for prefix, suffix in contexts
        ):
            raise ValidationError(f"action label {label!r} is not boundary-invariant")
        action_tokens.append(token)
    action_ids = {token.token_id for token in action_tokens}
    if len(action_ids) != len(action_tokens):
        raise ValidationError("action labels map to duplicate token IDs")
    overlap = action_ids & (fixed_ids | special_ids)
    if overlap:
        raise ValidationError(
            f"action token IDs overlap fixed/special prompt IDs: {sorted(overlap)}"
        )

    excluded = fixed_ids | special_ids | action_ids
    candidates: list[TokenSpec] = []
    for token_id in range(len(tokenizer)):
        if token_id in excluded:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not _state_candidate_text(text):
            continue
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            continue
        token = TokenSpec(text=text, token_id=token_id)
        contexts = (
            (CURRENT_STATE_PREFIX, "\n"),
            (GO_TO_FRAGMENT, COST_FRAGMENT + "1"),
        )
        if all(
            _single_token_in_context(tokenizer, token, prefix, suffix)
            for prefix, suffix in contexts
        ):
            candidates.append(token)

    candidates.sort(
        key=lambda token: _hash_rank(pool_seed, tokenizer_fingerprint, token.token_id)
    )
    if len(candidates) < state_pool_size:
        raise ValidationError(
            f"only {len(candidates)} safe state tokens found, requested {state_pool_size}"
        )

    registry = TokenRegistry(
        source=source,
        revision=revision,
        tokenizer_path=str(root),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_fingerprint=tokenizer_fingerprint,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        action_tokens=tuple(action_tokens),
        state_tokens=tuple(candidates[:state_pool_size]),
        fixed_token_ids=tuple(sorted(fixed_ids)),
        special_token_ids=tuple(sorted(special_ids)),
        pool_seed=pool_seed,
    )
    validate_token_registry(registry)

    # The next generated token must be the exact same action token used in the
    # input mask. This catches chat-template whitespace regressions.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Current State: test\nActions:\n"},
    ]
    rendered = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    for token in registry.action_tokens:
        completion_ids = tokenizer.encode(
            rendered + token.text, add_special_tokens=False
        )
        if completion_ids != prompt_ids + [token.token_id]:
            raise ValidationError(
                f"assistant completion boundary changes action token {token.text!r}"
            )
    return registry


def validate_token_registry(registry: TokenRegistry) -> None:
    if registry.schema_version != TOKEN_REGISTRY_SCHEMA_VERSION:
        raise ValidationError(
            f"unexpected token registry schema: {registry.schema_version}"
        )
    action_ids = {token.token_id for token in registry.action_tokens}
    state_ids = {token.token_id for token in registry.state_tokens}
    if len(action_ids) != len(registry.action_tokens):
        raise ValidationError("duplicate action token IDs in registry")
    if len(state_ids) != len(registry.state_tokens):
        raise ValidationError("duplicate state token IDs in registry")
    fixed_ids = set(registry.fixed_token_ids) | set(registry.special_token_ids)
    if action_ids & state_ids or action_ids & fixed_ids or state_ids & fixed_ids:
        raise ValidationError(
            "state, action, and fixed/special token ID sets must be disjoint"
        )
