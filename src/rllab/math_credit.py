"""CPU definitions for the bounded real-math prefix-value diagnostic."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import re
from typing import Any


INSTRUCTION = "Solve the following problem step by step. Put your final answer in \\boxed{} separated from your reasoning.\n\n"


def keyed_seed(seed: int, *parts: Any) -> int:
    key = json.dumps([seed, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(sha256(key.encode()).digest()[:8], "big") % (2**63 - 1)


def prompt_ids(tokenizer, problem: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + problem}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer(text: str) -> str | None:
    """Read the final box with balanced braces; do not guess from the last number."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                value = text[start:index].strip()
                return value or None
    return None


def grade_completion(text: str, target: str) -> dict:
    from rllab._vendor.prm800k.grader import grade_answer

    extracted = extract_answer(text)
    if extracted is None:
        return {
            "answer": None,
            "reward": 0,
            "grading_status": "missing_or_incomplete_box",
        }
    try:
        correct = bool(grade_answer(extracted, target))
    except Exception as error:
        return {
            "answer": extracted,
            "reward": 0,
            "grading_status": "grader_error",
            "error_type": type(error).__name__,
        }
    return {
        "answer": extracted,
        "reward": int(correct),
        "grading_status": "correct" if correct else "extracted_wrong",
    }


def trim_generation(ids: list[int], eos_ids: list[int]) -> tuple[list[int], bool]:
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[: index + 1], True
    return ids, False


def sampled_positions(root_ids: list[int], positions: list[int]) -> list[int]:
    # The last state is known under the finite token-budget environment.
    return sorted({p for p in positions if 0 <= p < len(root_ids)} | {len(root_ids)})


def wilson_interval(successes: int, n: int) -> list[float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid binomial observation")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def value_summary(rewards: list[int]) -> dict:
    if not rewards or len(rewards) % 2 or any(r not in (0, 1) for r in rewards):
        raise ValueError("Two equal, nonempty groups of binary outcomes are required")
    half = len(rewards) // 2
    successes = [sum(rewards[:half]), sum(rewards[half:])]
    values = [s / half for s in successes]
    return {
        "n": len(rewards),
        "successes": sum(successes),
        "value": sum(rewards) / len(rewards),
        "half_successes": successes,
        "half_values": values,
        "half_absolute_difference": abs(values[0] - values[1]),
        "wilson_95": wilson_interval(sum(successes), len(rewards)),
    }


def adjacent_advantages(states: list[dict]) -> list[dict]:
    ordered = sorted(states, key=lambda s: s["position"])
    return [
        {
            "from": left["position"],
            "to": right["position"],
            "advantage": right["value"] - left["value"],
            "half_advantages": [
                b - a for a, b in zip(left["half_values"], right["half_values"])
            ],
        }
        for left, right in zip(ordered, ordered[1:])
    ]


def process_token_credit(
    length: int, reward: int, baseline: float, prefixes: list[dict]
) -> dict:
    """Adjacent prefix values, directly broadcast to blocks including terminal.

    The initial leave-one-out baseline is shared between the MC halves: those
    halves diagnose suffix noise, not uncertainty in the initial baseline.
    """
    positions = [p["position"] for p in prefixes]
    assert length > 0 and positions == sorted(set(positions))
    assert all(0 < p < length for p in positions)
    states = [
        {"position": 0, "value": baseline, "half_values": [baseline, baseline]},
        *prefixes,
        {
            "position": length,
            "value": float(reward),
            "half_values": [float(reward)] * 2,
        },
    ]
    blocks = adjacent_advantages(states)
    tokens = [b["advantage"] for b in blocks for _ in range(b["to"] - b["from"])]
    assert len(tokens) == length
    return {"states": states, "blocks": blocks, "token_advantages": tokens}
