"""Online MC budgeting rules and independent receipt reconstruction."""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random

from rllab.math_alignment import active_question_batch, aggregate_capture
from rllab.math_io import fingerprint
from rllab.math_mc_cross_risk import allocate, credit_digest, question_contrast
from rllab.math_training import outcome_credits, training_root_jobs
from rllab.real_math_loop import assemble_credits, mc_prefix_counts

RULE = "online_cross_risk_pilot16_mixed32_v1"
TAG = "llama-math-mc-cross-risk24-16ppu-v2-20261012"
CONTRACT = "docs/training.md"
SPEC = dict(
    rule=RULE,
    pilot=16,
    maximum=32,
    average_ceiling=24,
    reference_roots=2,
    reference_order_seed=20261012,
)


def make_plan(original, devices=16):
    """Reconstruct the original 24-step phase (also used by historical resume)."""
    assert devices in (8, 16)
    expected = dict(
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
    )
    assert all(original["study"][k] == v for k, v in expected.items())
    assert len(original["train"]) == 1024 and len(original["evaluation"]) == 512
    plan = deepcopy(original)
    for k in ("created_at", "commit", "worktree_status"):
        plan.pop(k, None)
    plan["study"].update(
        name=TAG,
        updates=24,
        eval_every=8,
        checkpoint_every=4,
        evaluation_seed=20260922,
        max_wall_seconds=129600,
    )
    plan["schedule"] = plan["schedule"][:24]
    plan["evaluation"] = plan["train"] + original["evaluation"]
    plan["counts"].update(
        evaluation=1536, train_question_slots=384, distinct_scheduled_questions=384
    )
    plan["mc_cross_risk"] = dict(SPEC, devices=devices)
    assert all(
        training_root_jobs(plan, v) == training_root_jobs(original, v)
        for v in range(24)
    )
    return plan


def make_training_plan(original, mode="dynamic", devices=8):
    """Prepare a fresh 64-step run with paired roots and evaluation keys.

    All arms share the population, schedule, loss and update scale. Only the
    suffix allocation changes. Historical 24-to-64 continuation remains a
    separate, explicitly bound checkpoint path.
    """
    if mode not in ("dynamic", "uniform24", "uniform32"):
        raise ValueError(f"Unknown MC allocation: {mode}")
    plan = make_plan(original, devices=devices)
    plan["study"].update(name=f"llama-math-{mode}", updates=64)
    plan["schedule"] = deepcopy(original["schedule"])
    plan["counts"].update(train_question_slots=1024, distinct_scheduled_questions=1024)
    plan["historical_test_evaluation"] = dict(
        step=64,
        seed=20260914,
        samples=4,
        question_ids=[q["question_id"] for q in original["evaluation"]],
    )
    if mode != "dynamic":
        del plan["mc_cross_risk"]
        plan["study"]["mc_samples"] = 24 if mode == "uniform24" else 32
    else:
        validate_plan(plan, "prm", devices, 1)
    return plan


def capture_devices(plan):
    # The immutable v1 plans predate an explicit topology and used eight ranks.
    devices = plan["mc_cross_risk"].get("devices", 8)
    assert devices in (8, 16)
    return devices


def validate_plan(plan, credit, devices, nodes):
    assert credit == "prm" and (devices, nodes) == (capture_devices(plan), 1)
    assert {k: v for k, v in plan["mc_cross_risk"].items() if k != "devices"} == SPEC
    assert not any(
        k in plan for k in ("branch_training", "mc_allocation", "cross_task_trust")
    )
    s = plan["study"]
    updates = s["updates"]
    assert updates in (24, 64)
    if "continuation" in plan:
        from rllab.math_mc_cross_risk_continuation import validate_continuation

        validate_continuation(plan)
        updates = 64
    assert (
        s["updates"],
        s["questions_per_update"],
        s["roots_per_question"],
        s["mc_samples"],
        s["eval_every"],
        s["checkpoint_every"],
    ) == (updates, 16, 8, 32, 8, 4)
    assert (
        s["actor"],
        s["temperature"],
        s["scale_rule"],
        s["target_parameter_step_l2"],
    ) == ("llama", 0.6, "fixed_parameter_step", 0.01)
    assert len(set(sum(plan["schedule"], []))) == 16 * updates
    assert len(plan["schedule"]) == updates
    assert all(len(batch) == 16 for batch in plan["schedule"])
    assert len(plan["train"]) == 1024 and len(plan["evaluation"]) == 1536


def reference_groups(plan, version):
    ids = [q["question_id"] for q in plan["train"]]
    random.Random(SPEC["reference_order_seed"]).shuffle(ids)
    source = set(plan["schedule"][version])
    ids = [q for q in ids if q not in source]
    assert len(ids) == 1008 and len(set(ids)) == 1008
    return [ids[:504], ids[504:]]


def reference_batches(groups, rows):
    assert [(r["question_id"], r["root_index"]) for r in rows] == [
        (q, i) for group in groups for q in group for i in range(2)
    ]
    lookup = {(r["question_id"], r["root_index"]): r for r in rows}
    for gi, ids in enumerate(groups):
        for start in range(0, len(ids), 32):
            part = ids[start : start + 32]
            batch = [lookup[q, i] for q in part for i in range(2)]
            credit = outcome_credits(batch)
            scaled = [
                dict(
                    token_advantages=[
                        v * len(part) / len(ids) for v in c["token_advantages"]
                    ]
                )
                for c in credit
            ]
            yield f"fit{gi}", batch, scaled


def pilot_data(roots, pilot):
    credit = assemble_credits(roots, pilot, 16)
    pools = {}
    for row in pilot:
        key = row["root_slot"], row["position"]
        pools.setdefault(key, {})[row["sample_index"]] = row["reward"]
    return credit, {key: [v[i] for i in range(16)] for key, v in pools.items()}


def question_counts(roots):
    prefixes = mc_prefix_counts(roots, 16)
    counts = dict.fromkeys(dict.fromkeys(r["question_id"] for r in roots), 0)
    for slot, _ in prefixes:
        counts[roots[slot]["question_id"]] += 1
    return counts


def allocation(scores, counts):
    active = {q: n for q, n in counts.items() if n}
    assert set(scores) == set(counts)
    decision = allocate({q: scores[q] for q in active}, active)
    decision["budgets"] = {q: decision["budgets"].get(q, 16) for q in counts}
    return decision


def selected_counts(roots, decision):
    return {
        (slot, p): decision["budgets"][roots[slot]["question_id"]]
        for slot, p in mc_prefix_counts(roots, 16)
    }


def audit_selection(plan, roots, suffixes, references, record):
    """Rebuild risk/greedy allocation from scalar receipts and freshly graded rows."""
    devices = capture_devices(plan)
    folder, version = Path(record["attempt"]), record["step"] - 1
    ref = record["mc_cross_risk"]
    assert fingerprint(ref["path"]) == ref["sha256"]
    receipt = json.loads(Path(ref["path"]).read_text())
    assert receipt["version"] == version and receipt["rule"] == RULE
    groups = reference_groups(plan, version)
    assert receipt["reference_groups"] == groups
    assert all(r["role"] == "mc_reference" for r in references)
    if references:
        assert len(references) == 2016
    pilot = [r for r in suffixes if r["sample_index"] < 16]
    saved, pools = pilot_data(roots, pilot)
    counts = question_counts(roots)
    records = receipt["records"]
    by_name = {r["name"]: r for r in records}
    assert len(by_name) == len(records) <= 81

    def check(name, rows, credits):
        r = by_name[name]
        kept, _, scale = active_question_batch(rows, credits, multiple=devices)
        assert r["credit_digest"] == credit_digest(rows, credits)
        assert (r["roots"], r["computed_roots"], r["credit_scale"]) == (
            len(rows),
            len(kept),
            scale,
        )
        assert len(r["receipts"]) == devices
        c = aggregate_capture(r["receipts"])
        assert math.isclose(
            math.sqrt(c["norm2"]),
            r["native_gradient_norm"],
            rel_tol=1e-4,
            abs_tol=1e-10,
        )
        return c

    scores = dict.fromkeys(counts, 0.0)
    expected_names = []
    if any(counts.values()):
        assert len(references) == 2016
        for i, (_, rows, credits) in enumerate(reference_batches(groups, references)):
            check(f"target_{i}", rows, credits)
            expected_names.append(f"target_{i}")
        g = check("pilot", roots, saved)
        expected_names.append("pilot")
        assert receipt["fallback"] == (
            "zero_pilot_gradient" if not g["norm2"] else None
        )
        if g["norm2"]:
            for qi, q in enumerate(counts):
                if not counts[q]:
                    continue
                terms = []
                for k in range(3):
                    name = f"q{qi}_z{k}"
                    z = check(
                        name, roots, question_contrast(roots, saved, pools, q, 0, k)
                    )
                    expected_names.append(name)
                    for h in ("fit0", "fit1"):
                        terms.append(
                            (
                                0.01
                                / math.sqrt(g["norm2"])
                                * (
                                    z["dots"][h]
                                    - g["dots"][h] * z["dots"]["pilot"] / g["norm2"]
                                )
                            )
                            ** 2
                        )
                scores[q] = math.fsum(terms) / 6
        assert receipt["initial_signatures"] == receipt["final_signatures"]
        for key in ("begin", "release"):
            assert sorted(r["rank"] for r in receipt[key]) == list(range(devices))
        assert all(r["scheduler_epoch"] == version for r in receipt["begin"])
        for r in receipt["release"]:
            assert r["scheduler_epoch"] == version and r["intercepted_steps"] == len(
                records
            )
            assert r["scheduler_restored"] and r["rng_restored"]
    else:
        assert not references and receipt["fallback"] == "no_prefixes"
    assert [r["name"] for r in records] == expected_names
    assert receipt["scores"] == scores
    active = [q for q in counts if counts[q]]
    order = sorted(active, key=lambda q: (-scores[q] / counts[q], q))
    budgets = dict.fromkeys(counts, 16)
    calls, ceiling = 16 * sum(counts.values()), 24 * sum(counts.values())
    for q in order:
        if calls + 16 * counts[q] <= ceiling:
            budgets[q] = 32
            calls += 16 * counts[q]
    expected = dict(
        budgets=budgets,
        order=order,
        calls=calls,
        ceiling=ceiling,
        unused_calls=ceiling - calls,
    )
    assert receipt["allocation"] == expected
    phase_names = [f"v{version}-suffix-pilot", f"v{version}-suffix-topup"]
    assert record["suffixes_phases"] == phase_names and record["suffixes_phase"] is None
    events = json.loads((folder / "report.json").read_text())["events"]
    freeze = [
        i
        for i, e in enumerate(events)
        if e["stage"] == "mc_cross_risk_selection_frozen" and e["version"] == version
    ]
    topup = [
        i
        for i, e in enumerate(events)
        if e["stage"] == "sampling_started" and e["phase"] == phase_names[1]
    ]
    assert len(freeze) == len(topup) == 1 and freeze[0] < topup[0]
    actual = Counter((r["root_slot"], r["position"]) for r in suffixes)
    result = {
        (slot, p): budgets[roots[slot]["question_id"]]
        for slot, p in mc_prefix_counts(roots, 16)
    }
    assert actual == result and len(suffixes) == calls
    assert (ref["mc_calls"], ref["uniform32_calls"], ref["reference_calls"]) == (
        calls,
        32 * sum(counts.values()),
        len(references),
    )
    assert ref["selector_seconds"] == receipt["seconds"]
    return result
