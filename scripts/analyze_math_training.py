"""Reconstruct committed training records and compare fixed paired learning curves."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from rllab.math_credit import grade_completion, keyed_seed
from rllab.math_io import atomic_json, fingerprint, provenance, read_jsonl

GRID = [32, 64, 128, 256, 512, 1024, 2048]


def reconstruct_phase(
    folder, phase, plan, version, tokenizer, *, parents=None, regrade=True
):
    rows = read_jsonl(folder / f"{phase}.jsonl")
    grades = read_jsonl(folder / f"{phase}-grades.jsonl")
    questions = {q["question_id"]: q for q in plan["train"] + plan["evaluation"]}
    seen = set()
    for r, g in zip(rows, grades, strict=True):
        key = tuple(r[k] for k in ("root_slot", "position", "sample_index"))
        assert key not in seen and key == tuple(
            g[k] for k in ("root_slot", "position", "sample_index")
        )
        seen.add(key)
        assert r["weight_version"] == version
        assert all(
            r["extra_fields"][k] == version
            for k in ("global_steps", "min_global_steps", "max_global_steps")
        )
        ids, p = r["generated_ids"], r["position"]
        assert 0 < len(ids) <= 3072 - p and r["remaining_budget"] == 3072 - p
        assert not any(t in plan["eos"] for t in ids[:-1])
        assert ids[-1] in plan["eos"] or len(ids) == 3072 - p
        assert r["hit_token_cap"] == (ids[-1] not in plan["eos"])
        seed_version = 0 if r["role"] == "eval" else version
        seed_policy = plan["study"].get("training_seed_policy", "version")
        assert seed_policy in ("version", "question")
        if r["role"] == "root" and seed_policy == "question":
            # Independently reconstruct the declared single-exposure seed rule.
            slots = sum(plan["schedule"], [])
            assert len(slots) == len(set(slots))
            seed_version = 0
        seed = plan["study"]["seed"]
        if r["role"] == "eval":
            seed = plan["study"].get("evaluation_seed", seed)
        parts = [
            seed,
            "native-math",
            plan["study"].get("actor", "qwen"),
            seed_version,
            r["question_id"],
            r["role"],
            r["root_index"],
        ]
        prefix = []
        if parents is not None:
            parent = parents[r["root_slot"]]
            assert (
                parent["question_id"] == r["question_id"]
                and parent["root_index"] == r["root_index"]
            )
            assert p in GRID and p < len(parent["generated_ids"])
            prefix = parent["generated_ids"][:p]
            parts.extend([p, r["sample_index"]])
        else:
            assert p == 0 and r["sample_index"] == r["root_index"]
        assert r["seed"] == keyed_seed(*parts)
        if r["role"] == "root":
            assert len(r["log_probs"]) == len(ids) and all(
                math.isfinite(v) for v in r["log_probs"]
            )
        else:
            assert r["log_probs"] is None
        if regrade:
            text = tokenizer.decode(
                prefix + ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            expected = grade_completion(text, questions[r["question_id"]]["answer"])
            assert all(g[k] == v for k, v in expected.items())
        assert g["grading_status"] != "grader_error" and g["reward"] in (0, 1)
        r.update(g)
    return rows


def analyze_arm(root, credit, regrade_mc):
    from transformers import AutoTokenizer

    arm = root / credit
    plan = json.loads((arm / "plan.json").read_text())
    state = json.loads((arm / "state.json").read_text())
    s = plan["study"]
    assert fingerprint(arm / "plan.json") == state["plan_sha256"]
    assert state["status"] in ("paused", "completed")
    assert state["credit"] == credit
    assert [r["step"] for r in state["history"]] == list(range(1, state["step"] + 1))
    if "continuation" in plan:
        from rllab.math_mc_cross_risk_continuation import verify_state

        verify_state(plan, state)
    tokenizer = AutoTokenizer.from_pretrained(s["model"], local_files_only=True)
    totals = dict(root_tokens=0, mc_tokens=0, training_roots=0, mc_suffixes=0)
    if "mc_cross_risk" in plan:
        from rllab.math_mc_cross_risk_training import audit_selection, validate_plan

        validate_plan(
            plan,
            credit,
            state["topology"]["devices_per_node"],
            state["topology"]["nodes"],
        )
        totals.update(
            reference_roots=0,
            reference_tokens=0,
            uniform32_mc_calls=0,
            selector_seconds=0.0,
        )
    for record in state["history"]:
        folder, version = Path(record["attempt"]), record["step"] - 1
        roots = reconstruct_phase(
            folder, record["roots_phase"], plan, version, tokenizer
        )
        assert [r["question_id"] for r in roots] == [
            q for q in plan["schedule"][version] for _ in range(s["roots_per_question"])
        ]
        totals["root_tokens"] += sum(len(r["generated_ids"]) for r in roots)
        totals["training_roots"] += len(roots)
        values = {}
        if credit == "prm":
            process_ids = {r["question_id"] for r in roots}
            phases = (
                record["suffixes_phases"]
                if "mc_cross_risk" in plan
                else [record["suffixes_phase"]]
            )
            suffixes = []
            for phase_index, phase in enumerate(phases):
                phase_rows = reconstruct_phase(
                    folder,
                    phase,
                    plan,
                    version,
                    tokenizer,
                    parents=roots,
                    regrade=regrade_mc,
                )
                if "mc_cross_risk" in plan:
                    assert all(
                        (r["sample_index"] < 16) == (phase_index == 0)
                        for r in phase_rows
                    )
                suffixes.extend(phase_rows)
            pools = {}
            for r in suffixes:
                assert r["sample_index"] not in pools.get(
                    (r["root_slot"], r["position"]), {}
                )
                pools.setdefault((r["root_slot"], r["position"]), {})[
                    r["sample_index"]
                ] = r["reward"]
            assert set(pools) == {
                (i, p)
                for i, r in enumerate(roots)
                for p in GRID
                if r["question_id"] in process_ids
                if p < len(r["generated_ids"])
            }
            counts = dict.fromkeys(pools, s["mc_samples"])
            if "mc_cross_risk" in plan:
                references = reconstruct_phase(
                    folder,
                    record["mc_cross_risk"]["reference_phase"],
                    plan,
                    version,
                    tokenizer,
                )
                counts = audit_selection(plan, roots, suffixes, references, record)
                totals["reference_roots"] += len(references)
                totals["reference_tokens"] += sum(
                    len(r["generated_ids"]) for r in references
                )
                totals["uniform32_mc_calls"] += 32 * len(pools)
                totals["selector_seconds"] += record["mc_cross_risk"][
                    "selector_seconds"
                ]
            assert set(counts) == set(pools)
            assert all(
                set(pool) == set(range(counts[key])) for key, pool in pools.items()
            )
            values = {key: mean(pool.values()) for key, pool in pools.items()}
            totals["mc_tokens"] += sum(len(r["generated_ids"]) for r in suffixes)
            totals["mc_suffixes"] += len(suffixes)
        else:
            assert record["suffixes_phase"] is None
        loss_rows = roots
        credits = json.loads((folder / f"v{version}-credits.json").read_text())
        for i, (r, c) in enumerate(zip(loss_rows, credits, strict=True)):
            peers = [
                p["reward"]
                for j, p in enumerate(loss_rows)
                if j != i and p["question_id"] == r["question_id"]
            ]
            assert len(peers) == s["roots_per_question"] - 1
            positions = (
                [0] + [p for p in GRID if (i, p) in values] + [len(r["generated_ids"])]
            )
            state_values = (
                [mean(peers)] + [values[i, p] for p in positions[1:-1]] + [r["reward"]]
            )
            expected = [
                state_values[j + 1] - state_values[j]
                for j in range(len(positions) - 1)
                for _ in range(positions[j + 1] - positions[j])
            ]
            assert c["token_advantages"] == expected
            assert [v["position"] for v in c["states"]] == positions
            assert [v["value"] for v in c["states"]] == state_values
            for v in c["states"][1:-1]:
                pool = pools[i, v["position"]]
                half = len(pool) // 2
                halves = [sum(pool[k] for k in range(j, j + half)) for j in (0, half)]
                assert v["n"] == len(pool) and v["half_successes"] == halves
                assert v["half_values"] == [x / half for x in halves]
        assert record["metrics"]["lr"] == [s["learning_rate"]]
        applied_lr = record.get("effective_learning_rate", s["learning_rate"])
        grad_norm = record["metrics"]["grad_norm"][0]
        assert math.isfinite(applied_lr) and applied_lr > 0
        assert math.isclose(
            record["ideal_sgd_update_norm"], applied_lr * grad_norm, rel_tol=1e-5
        )
        if s.get("scale_rule", "fixed_learning_rate") == "fixed_parameter_step":
            target = s["target_parameter_step_l2"]
            assert target > 0 and math.isfinite(target) and grad_norm > 0
            assert math.isclose(applied_lr * grad_norm, target, rel_tol=1e-5)
            assert all(
                math.isclose(
                    a["step_scale"]["global_grad_norm"], grad_norm, rel_tol=1e-5
                )
                and math.isclose(
                    a["step_scale"]["nominal_step_l2"], target, rel_tol=1e-5
                )
                and math.isclose(
                    a["step_scale"]["effective_lr"], applied_lr, rel_tol=1e-6
                )
                and all(
                    math.isclose(lr, applied_lr, rel_tol=1e-6) for lr in a["applied_lr"]
                )
                and math.isfinite(a["sampled_bf16_update_norm"])
                for a in record["audit"]
            )
        else:
            assert s.get("scale_rule", "fixed_learning_rate") == "fixed_learning_rate"
            assert s.get("target_parameter_step_l2") is None
            assert applied_lr == s["learning_rate"]
        assert all(
            math.isfinite(record["metrics"][k][0]) for k in ("loss", "grad_norm")
        )
        assert all(
            a["optimizer_steps"] == 1
            and a["sample_finite"]
            and a["optimizer_state_entries"] == a["frozen_local_numel"] == 0
            for a in record["audit"]
        )
    curves = {}
    for key, record in state["evaluations"].items():
        for path, digest in record.get("shared_from", {}).get("files", {}).items():
            assert fingerprint(path) == digest
        step = int(key)
        rows = reconstruct_phase(
            Path(record["attempt"]), record["phase"], plan, step, tokenizer
        )
        assert [r["question_id"] for r in rows] == [
            q["question_id"]
            for q in plan["evaluation"]
            for _ in range(s["eval_samples"])
        ]
        by_q = {
            q["question_id"]: mean(
                r["reward"] for r in rows if r["question_id"] == q["question_id"]
            )
            for q in plan["evaluation"]
        }
        assert by_q == record["summary"]["by_question"]
        assert mean(by_q.values()) == record["summary"]["pass_at_1"]
        curves[key] = dict(pass_at_1=mean(by_q.values()), by_question=by_q)
    expected_steps = {str(v) for v in range(0, state["step"] + 1, s["eval_every"])}
    if state["step"] == s["updates"]:
        expected_steps.add(str(s["updates"]))
    assert set(curves) == expected_steps
    from rllab.math_mc_cross_risk_continuation import attempt_lineage

    attempts = []
    for path, expected_sha in attempt_lineage(plan, arm):
        attempt = json.loads(path.read_text())
        attempts.append(attempt)
        assert attempt["plan_sha256"] == expected_sha
        if attempt["resumed_from"]:
            verified = [
                e for e in attempt["events"] if e["stage"] == "restore_verified"
            ]
            assert len(verified) == 1
            marker = json.loads(
                (Path(attempt["resumed_from"]) / "complete.json").read_text()
            )
            assert verified[0]["signatures"] == marker["signatures"]
    historical_curves = {}
    if ("continuation" in plan or "historical_test_evaluation" in plan) and state[
        "status"
    ] == "completed":
        from rllab.math_mc_cross_risk_continuation import historical_evaluation_plan

        evaluation_plan = historical_evaluation_plan(plan)
        assert set(state["historical_evaluations"]) == {str(s["updates"])}
        record = state["historical_evaluations"][str(s["updates"])]
        assert record["phase"] == f"eval-historical-v{s['updates']}"
        rows = reconstruct_phase(
            Path(record["attempt"]),
            record["phase"],
            evaluation_plan,
            s["updates"],
            tokenizer,
        )
        ids = [q["question_id"] for q in evaluation_plan["evaluation"]]
        assert [r["question_id"] for r in rows] == [q for q in ids for _ in range(4)]
        by_q = {
            q: mean(r["reward"] for r in rows if r["question_id"] == q) for q in ids
        }
        assert by_q == record["summary"]["by_question"]
        assert mean(by_q.values()) == record["summary"]["pass_at_1"]
        historical_curves[str(s["updates"])] = dict(
            pass_at_1=mean(by_q.values()), by_question=by_q
        )
    return dict(
        credit=credit,
        status=state["status"],
        completed_updates=state["step"],
        plan_sha256=state["plan_sha256"],
        initial_signatures=state["initial_signatures"],
        counts=plan["counts"],
        totals=totals,
        curves=curves,
        historical_test_curves=historical_curves,
        study=s,
        active_runtime_seconds=sum(a["elapsed_seconds"] for a in attempts),
        allocated_device_hours=sum(a["elapsed_seconds"] for a in attempts)
        * state["topology"]["nodes"]
        * state["topology"]["devices_per_node"]
        / 3600,
        recorded_generation_tokens_all_attempts=sum(
            e["tokens"]
            for a in attempts
            for e in a["events"]
            if e["stage"] == "sampling_complete"
        ),
        cost_scope="Recorded phases and driver wall time across all attempts; unsaved interrupted work may be missing",
        mc_grades_independently_recomputed=regrade_mc,
        scope="Committed checkpoint lineage only; abandoned attempts remain separate evidence",
    )


def compare(orm, prm):
    import numpy as np

    assert orm["plan_sha256"] == prm["plan_sha256"]
    assert orm["initial_signatures"] == prm["initial_signatures"]
    assert orm["status"] == prm["status"] == "completed"
    assert not orm["study"]["acceptance"]
    steps = sorted(map(int, orm["curves"]))
    assert steps == sorted(map(int, prm["curves"]))
    assert steps[-1] == orm["study"]["updates"]
    ids = sorted(orm["curves"]["0"]["by_question"])
    arrays = {
        a["credit"]: np.array(
            [[a["curves"][str(t)]["by_question"][q] for q in ids] for t in steps]
        )
        for a in (orm, prm)
    }

    def auc(y):
        return (
            sum(
                (y[i] + y[i - 1]) * (steps[i] - steps[i - 1]) / 2
                for i in range(1, len(steps))
            )
            / steps[-1]
        )

    rng = np.random.default_rng(keyed_seed(orm["study"]["seed"], "question-bootstrap"))
    index = rng.integers(0, len(ids), size=(2000, len(ids)))
    differences = dict(
        endpoint=arrays["prm"][-1] - arrays["orm"][-1],
        improvement=(arrays["prm"][-1] - arrays["prm"][0])
        - (arrays["orm"][-1] - arrays["orm"][0]),
        normalized_auc=auc(arrays["prm"]) - auc(arrays["orm"]),
    )
    return dict(
        per_arm={
            credit: dict(
                initial_pass_at_1=float(y[0].mean()),
                endpoint_pass_at_1=float(y[-1].mean()),
                improvement=float((y[-1] - y[0]).mean()),
                normalized_auc=float(auc(y).mean()),
            )
            for credit, y in arrays.items()
        },
        prm_minus_orm={
            key: dict(
                estimate=float(values.mean()),
                question_bootstrap_95=np.quantile(
                    values[index].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            )
            for key, values in differences.items()
        },
        uncertainty="Paired test-question bootstrap; one training seed, no seed-variance estimate",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--credit", choices=["orm", "prm"])
    parser.add_argument("--regrade-mc", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    arms = {
        credit: analyze_arm(args.run_dir, credit, args.regrade_mc)
        for credit in ([args.credit] if args.credit else ["orm", "prm"])
    }
    result = dict(status="completed", **provenance(), arms=arms)
    if not args.credit:
        result["comparison"] = compare(arms["orm"], arms["prm"])
    atomic_json(args.output, result)
    print(json.dumps(dict(status="completed", output=str(args.output))))


if __name__ == "__main__":
    main()
