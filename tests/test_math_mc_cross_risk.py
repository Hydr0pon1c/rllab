"""Budget accounting, pilot separation and normalized influence algebra."""

import copy

import numpy as np
import pytest

from rllab.math_credit import process_token_credit, value_summary
from rllab.math_mc_cross_risk import (
    DELTA,
    allocate,
    mixed_credits,
    question_contrast,
    risk,
)


def test_normalized_risk_matches_finite_difference_and_removes_radial_noise():
    g = np.array([2.0, 1.0, -1.0])
    hs = [np.array([1.0, 2.0, 3.0]), np.array([2.0, -2.0, 1.0])]
    zs = [np.array([1.0, -1.0, 0.0]), np.array([0.0, 1.0, 1.0]), g.copy()]
    base = dict(
        norm2=float(g @ g), dots={str(i): float(h @ g) for i, h in enumerate(hs)}
    )
    contrasts = [
        dict(
            dots={
                "pilot": float(g @ z),
                **{str(i): float(h @ z) for i, h in enumerate(hs)},
            }
        )
        for z in zs
    ]
    epsilon = 1e-5
    differences = [
        float(
            h
            @ (
                DELTA * (g + epsilon * z) / np.linalg.norm(g + epsilon * z)
                - DELTA * (g - epsilon * z) / np.linalg.norm(g - epsilon * z)
            )
            / (2 * epsilon)
        )
        for z in zs
        for h in hs
    ]
    assert risk(base, contrasts, ["0", "1"]) == pytest.approx(
        np.mean(np.square(differences)), rel=1e-8
    )
    assert risk(base, [contrasts[-1]] * 3, ["0", "1"]) == pytest.approx(0, abs=1e-30)


def test_budget_ranks_by_risk_per_call_and_counts_pilot_once():
    counts = dict(a=10, b=20, c=30)
    chosen = allocate(dict(a=2.0, b=1.0, c=9.0), counts)
    assert chosen["order"] == ["c", "a", "b"]
    assert chosen["budgets"] == dict(a=16, b=16, c=32)
    assert chosen["calls"] == chosen["ceiling"] == 24 * 60
    assert sum(chosen["budgets"][q] * n for q, n in counts.items()) == chosen["calls"]
    assert allocate(dict(a=1.0, b=1.0, c=1.0), dict(a=1, b=1, c=1))["unused_calls"] == 8


def test_pilot_contrasts_ignore_audit_and_reused_credit_preserves_endpoints():
    rows = [dict(question_id=q, root_index=0) for q in ("a", "b")]
    pools = {(0, 2): [0] * 4 + [1] * 4 + [0] * 8 + [1] * 16, (1, 2): [1, 0] * 16}
    saved = [
        dict(
            root_slot=i,
            **process_token_credit(
                4, i, 0.25, [dict(position=2, **value_summary(pools[i, 2]))]
            ),
        )
        for i in range(2)
    ]
    altered = copy.deepcopy(pools)
    altered[0, 2][16:] = [0] * 16
    for k in range(3):
        before = question_contrast(rows, saved, pools, "a", 0, k)
        assert before == question_contrast(rows, saved, altered, "a", 0, k)
        assert not any(before[1]["token_advantages"])
        assert before[0]["token_advantages"][0] + before[0]["token_advantages"][-1] == 0
    low = mixed_credits(rows, saved, pools, dict(a=16, b=32))
    assert low == mixed_credits(rows, saved, altered, dict(a=16, b=32))
    high = mixed_credits(rows, saved, pools, dict(a=32, b=16))
    assert high != mixed_credits(rows, saved, altered, dict(a=32, b=16))
    for i, c in enumerate(low):
        assert c["token_advantages"][0] + c["token_advantages"][-1] == i - 0.25
