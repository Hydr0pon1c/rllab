"""The versioned one-turn PuzzleMDP prompt contract."""

from __future__ import annotations

PROMPT_TEMPLATE_VERSION = "qwen-puzzle-one-turn/v1"

SYSTEM_PROMPT = (
    "You are in a deterministic puzzle. At each state, choose one listed action "
    "to minimize the total cost until an exit. Reply with exactly one action token "
    "and nothing else."
)

CURRENT_STATE_PREFIX = "Current State: "
ACTIONS_HEADER = "Actions:\n"
GO_TO_FRAGMENT = " Go to "
COST_FRAGMENT = " Cost:"


def fixed_text_fragments() -> tuple[str, ...]:
    """Fragments whose token IDs must not overlap state/action token IDs."""

    return (
        SYSTEM_PROMPT,
        CURRENT_STATE_PREFIX,
        "\n",
        ACTIONS_HEADER,
        GO_TO_FRAGMENT,
        COST_FRAGMENT,
        *tuple(str(value) for value in range(10)),
    )
