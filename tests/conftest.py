from __future__ import annotations

from pathlib import Path
from importlib.util import find_spec

import pytest


@pytest.fixture
def real_rl_stack():
    missing = [name for name in ("torch", "ray", "verl") if find_spec(name) is None]
    if missing:
        pytest.skip(
            f"Native RL dependencies unavailable: {', '.join(missing)}; "
            "run with scripts/real_rl_python.sh"
        )


@pytest.fixture(scope="session")
def qwen_model_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "models" / "Qwen3-8B"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("local Qwen3-8B tokenizer snapshot is unavailable")
    return path.resolve()


@pytest.fixture(scope="session")
def qwen_tokenizer(qwen_model_path):
    from rllab.data.tokens import load_tokenizer

    return load_tokenizer(qwen_model_path)


@pytest.fixture(scope="session")
def qwen_registry(qwen_model_path, qwen_tokenizer):
    from rllab.data.tokens import build_token_registry

    return build_token_registry(
        qwen_tokenizer,
        tokenizer_path=qwen_model_path,
        source="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        action_labels=("A", "B", "C", "D"),
        state_pool_size=64,
        pool_seed=20260824,
        enable_thinking=False,
    )
