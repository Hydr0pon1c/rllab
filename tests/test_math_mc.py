"""Checks on the scientific boundaries of prefix-value measurement."""

from pathlib import Path

import pytest

from rllab.math_credit import (
    adjacent_advantages,
    extract_answer,
    grade_completion,
    sampled_positions,
    trim_generation,
    value_summary,
    prompt_ids,
)


def test_final_box_and_equivalent_answer():
    pytest.importorskip("pylatexenc")
    assert (
        extract_answer(r"work \boxed{3}; finally \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    )
    assert grade_completion(r"\boxed{\frac{1}{2}}", "0.5")["reward"] == 1
    assert grade_completion(r"\boxed{2}", "3")["reward"] == 0
    assert extract_answer(r"earlier \boxed{3}; final \boxed{\frac{1}{2}") is None
    assert (
        grade_completion("The calculation mentions 42", "42")["grading_status"]
        == "missing_or_incomplete_box"
    )


def test_mc_keeps_soft_values_and_terminal_reward():
    zero = value_summary([0] * 16)
    mixed = value_summary([0] * 8 + [1] * 8)
    assert zero["wilson_95"][1] > 0.19
    assert mixed["value"] == 0.5
    assert mixed["half_values"] == [0.0, 1.0]
    states = [
        {"position": 0, **mixed},
        {"position": 64, **zero},
        {"position": 70, "value": 1.0, "half_values": [1.0, 1.0]},
    ]
    advantages = adjacent_advantages(states)
    assert [a["advantage"] for a in advantages] == [-0.5, 1.0]
    assert sum(a["advantage"] for a in advantages) == 1 - mixed["value"]


def test_terminal_tokens_and_causal_positions():
    tokens, ended = trim_generation([4, 7, 2, 0, 0], [2, 3])
    assert tokens == [4, 7, 2] and ended
    assert sampled_positions(tokens, [0, 64, 128]) == [0, 3]
    assert sampled_positions(list(range(128)), [0, 64, 128]) == [0, 64, 128]


def test_local_qwen_template_returns_raw_token_ids():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    model_path = Path(__file__).resolve().parents[1] / "models/Qwen2.5-7B-Instruct"
    if not (model_path / "tokenizer.json").is_file():
        pytest.skip("Pinned local tokenizer is not present")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    ids = prompt_ids(tokenizer, "What is 1 + 1?")
    assert isinstance(ids, list) and all(isinstance(token, int) for token in ids)
    text = tokenizer.decode(ids)
    assert "What is 1 + 1?" in text
    assert text.endswith("<|im_start|>assistant\n")
