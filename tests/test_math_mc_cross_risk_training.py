"""Pilot retention, degenerate prefixes and matched original training schedule."""

from copy import deepcopy

import pytest

from rllab.math_mc_cross_risk_training import (
    allocation,
    make_plan,
    make_training_plan,
    pilot_data,
    question_counts,
    reference_groups,
    selected_counts,
    validate_plan,
)
from rllab.real_math_loop import assemble_credits, suffix_jobs


def test_reused_pilot_budget_zeros_and_exact_credit():
    roots = [
        dict(
            question_id=q, root_index=i, generated_ids=list(range(length)), reward=i % 2
        )
        for q, length in [("short", 4), ("long", 65), ("medium", 33)]
        for i in range(8)
    ]
    counts = question_counts(roots)
    assert counts == dict(short=0, long=16, medium=8)
    decision = allocation(dict(short=0.0, long=10.0, medium=1.0), counts)
    assert decision["calls"] <= 24 * 24
    assert decision["budgets"] == dict(short=16, long=16, medium=32)
    # Whole-question cost can skip a high-risk question that cannot fit.
    jobs = suffix_jobs(roots, {q: [1] for q in counts}, "llama", 0, 32)
    rows = [dict(j, reward=(j["sample_index"] + j["root_slot"]) % 3 == 0) for j in jobs]
    pilot = [r for r in rows if r["sample_index"] < 16]
    saved, pools = pilot_data(roots, pilot)
    assert all(len(v) == 16 for v in pools.values())
    suffixes = pilot + [
        r
        for r in rows
        if r["sample_index"] >= 16 and decision["budgets"][r["question_id"]] == 32
    ]
    assert len(suffixes) == decision["calls"]
    credits = assemble_credits(
        roots, suffixes, 32, prefix_counts=selected_counts(roots, decision)
    )
    for c, old in zip(credits, saved, strict=True):
        assert (
            c["states"][0] == old["states"][0] and c["states"][-1] == old["states"][-1]
        )
        if decision["budgets"][c["question_id"]] == 16:
            assert c == old
    assert allocation({"short": 0.0}, {"short": 0})["calls"] == 0


def test_plan_preserves_roots_and_references_exclude_only_current_batch():
    train = [dict(question_id=str(i)) for i in range(1024)]
    original = dict(
        study=dict(
            actor="llama",
            seed=20260914,
            updates=64,
            questions_per_update=16,
            roots_per_question=8,
            mc_samples=32,
            temperature=0.6,
            scale_rule="fixed_parameter_step",
            target_parameter_step_l2=0.01,
            learning_rate=0.0001,
        ),
        train=train,
        evaluation=[dict(question_id=f"t{i}") for i in range(512)],
        schedule=[[str(q) for q in range(i * 16, (i + 1) * 16)] for i in range(64)],
        prompts={str(i): [1] for i in range(1024)},
        counts={},
    )
    before = deepcopy(original)
    plan = make_plan(original)
    validate_plan(plan, "prm", 16, 1)
    with pytest.raises(AssertionError):
        validate_plan(plan, "prm", 8, 1)
    legacy = deepcopy(plan)
    del legacy["mc_cross_risk"]["devices"]
    validate_plan(legacy, "prm", 8, 1)
    assert original == before and plan["schedule"] == original["schedule"][:24]
    for v in (0, 23):
        groups = reference_groups(plan, v)
        assert list(map(len, groups)) == [504, 504]
        assert not set(sum(groups, [])) & set(plan["schedule"][v])
    # Public fresh-run recipes keep the complete source schedule and sampling
    # keys paired across dynamic and uniform allocations, including step 63.
    from rllab.math_training import training_root_jobs
    from rllab.math_mc_cross_risk_continuation import historical_evaluation_plan

    for mode, count in (("dynamic", 32), ("uniform24", 24), ("uniform32", 32)):
        fresh = make_training_plan(original, mode)
        assert fresh["schedule"] == original["schedule"]
        assert fresh["study"]["mc_samples"] == count
        assert fresh["evaluation"] == original["train"] + original["evaluation"]
        assert historical_evaluation_plan(fresh)["evaluation"] == original["evaluation"]
        for v in (0, 23, 63):
            assert training_root_jobs(fresh, v) == training_root_jobs(original, v)
        if mode == "dynamic":
            validate_plan(fresh, "prm", 8, 1)
        else:
            assert "mc_cross_risk" not in fresh
    assert original == before
