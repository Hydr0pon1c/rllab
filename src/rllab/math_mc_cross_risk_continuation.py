"""Explicit immutable lineage for the authorized mixed24 -> mixed64 extension."""

from copy import deepcopy
import json
from pathlib import Path

from rllab.math_io import fingerprint

TAG = "llama-math-mc-cross-risk64-16ppu-v1-20261013"
CONTRACT = "docs/training.md"
START = 24
END = 64


def load_bound(ref):
    assert fingerprint(ref["path"]) == ref["sha256"], ref["path"]
    return json.loads(Path(ref["path"]).read_text())


def scientific(plan):
    return {
        k: v
        for k, v in plan.items()
        if k not in ("created_at", "commit", "worktree_status")
    }


def make_plan(original, parent, inputs):
    from rllab.math_mc_cross_risk_training import make_plan as make_parent

    assert scientific(parent) == make_parent(original, devices=16), (
        "Continuation must retain the completed mixed24 scientific controls"
    )
    plan = deepcopy(scientific(parent))
    plan["study"].update(name=TAG, updates=END)
    plan["schedule"] = deepcopy(original["schedule"])
    assert len(plan["schedule"]) == END
    assert plan["schedule"][:START] == parent["schedule"]
    plan["counts"].update(train_question_slots=1024, distinct_scheduled_questions=1024)
    plan["continuation"] = dict(
        rule="mixed24_to64_v1",
        start_step=START,
        inputs=deepcopy(inputs),
        historical_test=dict(step=END, seed=20260914, samples=4),
    )
    return plan


def validate_continuation(plan):
    """Check only small immutable records; no tensor loads or parent writes."""
    from rllab.math_training import verify_resume

    spec = plan["continuation"]
    refs = spec["inputs"]
    assert set(refs) == {
        "original_plan",
        "parent_plan",
        "parent_state",
        "marker",
        "audit",
    }
    original, parent, state, marker, audit = (
        load_bound(refs[k])
        for k in ("original_plan", "parent_plan", "parent_state", "marker", "audit")
    )
    assert scientific(plan) == make_plan(original, parent, refs)
    assert state["status"] == "completed" and state["step"] == START
    assert state["topology"] == dict(nodes=1, devices_per_node=16)
    verify_resume(state, refs["parent_plan"]["sha256"], "prm", state["topology"])
    assert Path(refs["marker"]["path"]) == Path(state["checkpoint"]) / "complete.json"
    assert marker["step"] == START
    assert set(state["evaluations"]) == {"0", "8", "16", "24"}
    assert audit["status"] == "completed" and audit["completed_updates"] == START
    assert audit["mc_grades_independently_recomputed"] is True
    assert audit["plan_sha256"] == refs["parent_plan"]["sha256"]
    return state


def initial_state(plan, plan_sha, credit, topology):
    parent = validate_continuation(plan)
    assert credit == "prm" and topology == parent["topology"]
    checkpoint = Path(parent["checkpoint"])
    for rank in range(16):
        for kind in ("model", "optim", "extra_state"):
            payload = checkpoint / f"{kind}_world_size_16_rank_{rank}.pt"
            assert payload.is_file() and payload.stat().st_size > 0, str(payload)
    state = deepcopy(parent)
    state.update(
        status="running",
        plan_sha256=plan_sha,
        # Parent checkpoints must never enter this child's rotation list.
        checkpoints=[],
        continuation=deepcopy(plan["continuation"]),
    )
    return state


def verify_state(plan, state):
    parent = validate_continuation(plan)
    assert state["continuation"] == plan["continuation"]
    assert START <= state["step"] <= END
    assert state["history"][:START] == parent["history"]
    assert state["initial_signatures"] == parent["initial_signatures"]
    for step, record in parent["evaluations"].items():
        assert state["evaluations"][step] == record
    assert not set(state["checkpoints"]) & set(parent["checkpoints"])
    if state["step"] == START:
        assert state["checkpoint"] == parent["checkpoint"]
        assert state["checkpoints"] == []
        return plan["continuation"]["inputs"]["parent_plan"]["sha256"]
    assert state["checkpoint"] != parent["checkpoint"]
    return state["plan_sha256"]


def historical_evaluation_plan(plan):
    """Keep comparison evaluation keys separate from the uninterrupted main curve."""
    comparison = deepcopy(plan)
    if "continuation" in plan:
        spec = plan["continuation"]["historical_test"]
        original = load_bound(plan["continuation"]["inputs"]["original_plan"])
        comparison["evaluation"] = original["evaluation"]
    else:
        spec = plan["historical_test_evaluation"]
        by_id = {q["question_id"]: q for q in plan["evaluation"]}
        assert len(spec["question_ids"]) == len(set(spec["question_ids"]))
        comparison["evaluation"] = [by_id[q] for q in spec["question_ids"]]
    assert spec["step"] == plan["study"]["updates"] and spec["samples"] == 4
    comparison["study"].update(
        evaluation_seed=spec["seed"], eval_samples=spec["samples"]
    )
    return comparison


def attempt_lineage(plan, arm):
    """Both generations, without copying or editing any parent report."""
    paths = [
        (p, fingerprint(arm / "plan.json"))
        for p in sorted(arm.glob("attempts/*/report.json"))
    ]
    if "continuation" in plan:
        validate_continuation(plan)
        refs = plan["continuation"]["inputs"]
        parent_arm = Path(refs["parent_state"]["path"]).parent
        paths = [
            (p, refs["parent_plan"]["sha256"])
            for p in sorted(parent_arm.glob("attempts/*/report.json"))
        ] + paths
    return paths
