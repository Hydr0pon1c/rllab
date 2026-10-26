"""Plans, credit adapters and committed progress for paired native math training."""

from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean

from rllab.math_credit import keyed_seed, process_token_credit, prompt_ids
from rllab.math_io import fingerprint, now, provenance, read_jsonl

MODEL_REVISIONS = {
    "qwen": "a09a35458c702b33eeacc393d103063234e8bc28",
    "llama": "0e9e39f249a16976918f6564b8830bc894c89659",
    "gemma3": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
}


def actor_policy(study):
    """Old immutable Qwen plans retain their original seed namespace and T=1."""
    actor = study.get("actor", "qwen")
    if actor != "qwen" and "temperature" not in study:
        raise ValueError("A new actor requires an explicit sampling/loss temperature")
    temperature = float(study.get("temperature", 1.0))
    if (
        actor not in MODEL_REVISIONS
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid actor or sampling/loss temperature")
    return actor, temperature


def validate_evaluation_rows(rows, plan, questions, *, seed, version):
    """Fixed four-root evaluation keys are independent of the weight version."""
    actor, _ = actor_policy(plan["study"])
    assert [(r["question_id"], r["root_index"]) for r in rows] == [
        (q["question_id"], i) for q in questions for i in range(4)
    ]
    for slot, row in enumerate(rows):
        assert row["role"] == "eval" and row["root_slot"] == slot
        assert row["position"] == 0 and row["sample_index"] == row["root_index"]
        assert row["weight_version"] == version
        assert row["extra_fields"]["min_global_steps"] == version
        assert row["extra_fields"]["max_global_steps"] == version
        assert row["seed"] == keyed_seed(
            seed, "native-math", actor, 0, row["question_id"], "eval", row["root_index"]
        )
        tokens = row["generated_ids"]
        assert row["remaining_budget"] == 3072 and 0 < len(tokens) <= 3072
        assert not any(t in plan["eos"] for t in tokens[:-1])
        assert tokens[-1] in plan["eos"] or len(tokens) == 3072
        assert row["hit_token_cap"] == (tokens[-1] not in plan["eos"])
        assert row["reward"] in (0, 1)


def parameter_step_target(study):
    """Validate the learning rate or fixed parameter-step convention."""
    rule = study.get("scale_rule", "fixed_learning_rate")
    target = study.get("target_parameter_step_l2")
    if rule == "fixed_learning_rate" and target is None:
        return None
    if rule == "fixed_parameter_step" and target is not None:
        target = float(target)
        if math.isfinite(target) and target > 0:
            return target
    raise ValueError("Invalid scale_rule / target_parameter_step_l2 combination")


def question_schedule(questions, batch_size, updates, seed):
    ids = [q["question_id"] for q in questions]
    assert len(ids) == len(set(ids)) and len(ids) >= batch_size
    epoch, cursor, ordered, batch, schedule = 0, 0, [], [], []
    while len(schedule) < updates:
        if cursor == len(ordered):
            ordered = sorted(
                ids, key=lambda q: keyed_seed(seed, "train-order", epoch, q)
            )
            epoch, cursor = epoch + 1, 0
        qid = ordered[cursor]
        cursor += 1
        if qid not in batch:
            batch.append(qid)
        if len(batch) == batch_size:
            schedule.append(batch)
            batch = []
    return schedule


def training_root_jobs(plan, version):
    """Pair single-exposure composition arms by question; preserve old keys."""
    from rllab.real_math_loop import root_jobs

    study = plan["study"]
    policy = study.get("training_seed_policy", "version")
    if policy == "question":
        slots = sum(plan["schedule"], [])
        if len(slots) != len(set(slots)):
            raise ValueError("Question-paired seeds require one exposure per question")
        seed_version = 0
    elif policy == "version":
        seed_version = version
    else:
        raise ValueError(f"Unknown training_seed_policy: {policy}")
    actor, _ = actor_policy(study)
    by_id = {q["question_id"]: q for q in plan["train"]}
    return root_jobs(
        [by_id[qid] for qid in plan["schedule"][version]],
        plan["prompts"],
        actor,
        seed_version,
        study["roots_per_question"],
        seed=study["seed"],
    )


def build_plan(study, tokenizer, *, reference=None):
    parameter_step_target(study)
    actor, _ = actor_policy(study)
    source = study.get("source", "math")
    if source not in ("math", "dapo") or (
        source == "dapo" and study.get("level") is not None
    ):
        raise ValueError("Select MATH level or DAPO without a MATH level filter")
    rows = read_jsonl(study["dataset"])
    candidates = [
        q
        for q in rows
        if q["source"] == source and (source != "math" or q["level"] == study["level"])
    ]
    assert not (
        {q["group_id"] for q in candidates if q["split"] == "train"}
        & {q["group_id"] for q in candidates if q["split"] == "test"}
    )
    pools = {"train": [], "test": []}
    prompts, excluded, seen = {}, [], set()
    for q in sorted(candidates, key=lambda q: q["question_id"]):
        reason = None
        if "[asy]" in q["problem"] or any(
            ord(c) < 32 and c not in "\n\r\t" for c in q["problem"]
        ):
            reason = "unsupported_problem_text"
        elif q["group_id"] in seen:
            reason = "duplicate_question_family"
        if reason is None:
            ids = prompt_ids(tokenizer, q["problem"])
            if len(ids) > 1024:
                reason = "prompt_over_1024"
        if reason:
            excluded.append(dict(question_id=q["question_id"], reason=reason))
        else:
            seen.add(q["group_id"])
            pools[q["split"]].append(q)
            prompts[q["question_id"]] = ids
    assert not (
        {q["group_id"] for q in pools["train"]} & {q["group_id"] for q in pools["test"]}
    )
    train_pool_size = len(pools["train"])
    if reference is None:
        pools["train"] = sorted(
            pools["train"],
            key=lambda q: keyed_seed(study["seed"], "train-subset", q["question_id"]),
        )[: study["train_questions"]]
    else:
        assert not study["acceptance"] and not reference["study"]["acceptance"]
        assert reference["source_dataset_sha256"] == fingerprint(study["dataset"])
        for key in (
            "seed",
            "questions_per_update",
            "updates",
            "train_questions",
            "test_questions",
        ):
            assert study[key] == reference["study"][key], key
        # Reference IDs are mandatory: an incompatible native prompt must not
        # silently produce a different dataset in the second actor.
        eligible = {q["question_id"]: q for q in pools["train"]}
        assert all(eligible.get(q["question_id"]) == q for q in reference["train"]), (
            "Reference training questions fail this actor/source eligibility"
        )
        pools["train"] = reference["train"]
    assert len(pools["train"]) == study["train_questions"]
    schedule = question_schedule(
        pools["train"], study["questions_per_update"], study["updates"], study["seed"]
    )
    evaluation_pool = pools["train"] if study["acceptance"] else pools["test"]
    evaluation = sorted(
        evaluation_pool,
        key=lambda q: keyed_seed(study["seed"], "eval-order", q["question_id"]),
    )[: study["test_questions"]]
    assert len(evaluation) == study["test_questions"]
    if reference is not None:
        eligible = {q["question_id"]: q for q in pools["test"]}
        assert all(
            eligible.get(q["question_id"]) == q for q in reference["evaluation"]
        ), "Reference test questions fail this actor/source eligibility"
        evaluation = reference["evaluation"]
        assert schedule == reference["schedule"]
    selected = {q["question_id"] for q in pools["train"] + evaluation}
    native = json.loads((Path(study["model"]) / "generation_config.json").read_text())
    eos = native["eos_token_id"]
    return dict(
        created_at=now(),
        **provenance(),
        study=study,
        source_dataset_sha256=fingerprint(study["dataset"]),
        source_dataset_manifest=str(Path(study["dataset"]).with_name("manifest.json")),
        native_model_revision=MODEL_REVISIONS[actor],
        train=pools["train"],
        evaluation=evaluation,
        schedule=schedule,
        prompts={qid: ids for qid, ids in prompts.items() if qid in selected},
        eos=eos if isinstance(eos, list) else [eos],
        excluded=excluded,
        counts=dict(
            train=len(pools["train"]),
            train_pool=train_pool_size,
            test_pool=len(pools["test"]),
            evaluation=len(evaluation),
            train_question_slots=sum(map(len, schedule)),
            distinct_scheduled_questions=len(set(sum(schedule, []))),
        ),
    )


def repeat_plan(plan, seed):
    """Change training randomness without reselecting questions or monitoring keys."""
    if plan["study"]["acceptance"]:
        raise ValueError("Repeat a formal plan, not a startup acceptance plan")
    if type(seed) is not int or seed < 0 or seed == plan["study"]["seed"]:
        raise ValueError("A repeat requires a different nonnegative integer seed")
    result = deepcopy(plan)
    study = result["study"]
    study["evaluation_seed"] = study.get("evaluation_seed", study["seed"])
    study["name"] = f"{study['name']}-s{seed}"
    study["seed"] = seed
    schedule = question_schedule(
        result["train"], study["questions_per_update"], study["updates"], seed
    )
    result.update(created_at=now(), **provenance(), schedule=schedule)
    result["counts"].update(
        train_question_slots=sum(map(len, schedule)),
        distinct_scheduled_questions=len(set(sum(schedule, []))),
    )
    return result


def validate_budget_only_plan(parent, plan):
    """Accept only an operational wall-limit amendment, preserving every control."""

    def scientific(value):
        result = dict(value)
        result.pop("budget_amendment", None)
        result["study"] = dict(result["study"])
        result["study"].pop("max_wall_seconds", None)
        return result

    assert scientific(parent) == scientific(plan), "Scientific plan fields differ"


def validate_repeat(parent, plan):
    """Check the recorded seed-1001/1002 repeats without changing their controls."""
    for key in (
        "train",
        "evaluation",
        "prompts",
        "eos",
        "native_model_revision",
        "source_dataset_sha256",
    ):
        assert plan[key] == parent[key], key
    expected = dict(parent["study"])
    seed = plan["study"]["seed"]
    assert seed in (1001, 1002)
    expected.update(
        name=f"{expected['name']}-s{seed}",
        seed=seed,
        evaluation_seed=parent["study"].get("evaluation_seed", parent["study"]["seed"]),
    )
    assert plan["study"] == expected


def startup_plan(plan):
    """One full-size update and train-only evaluations, with no test exposure."""
    study = {
        **plan["study"],
        "name": plan["study"]["name"] + "-startup",
        "acceptance": True,
        "updates": 1,
        "eval_every": 1,
        "checkpoint_every": 1,
        "max_wall_seconds": 7200,
        "test_questions": plan["study"]["questions_per_update"],
    }
    by_id = {q["question_id"]: q for q in plan["train"]}
    batch = [by_id[qid] for qid in plan["schedule"][0]]
    return {
        **plan,
        "created_at": now(),
        **provenance(),
        "study": {**study, "train_questions": len(batch)},
        "train": batch,
        "evaluation": batch,
        "schedule": [plan["schedule"][0]],
        "prompts": {q["question_id"]: plan["prompts"][q["question_id"]] for q in batch},
        "counts": {
            **plan["counts"],
            "train": len(batch),
            "evaluation": len(batch),
            "train_question_slots": len(batch),
            "distinct_scheduled_questions": len(batch),
        },
    }


def outcome_credits(roots):
    rewards = defaultdict(list)
    for r in roots:
        rewards[r["question_id"]].append(r["reward"])
    credits = []
    for i, r in enumerate(roots):
        peers = rewards[r["question_id"]]
        assert len(peers) > 1
        baseline = (sum(peers) - r["reward"]) / (len(peers) - 1)
        credits.append(
            dict(
                question_id=r["question_id"],
                root_slot=i,
                root_index=r["root_index"],
                **process_token_credit(
                    len(r["generated_ids"]), r["reward"], baseline, []
                ),
            )
        )
    return credits


def completion_summary(rows):
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row["reward"])
    rates = {qid: mean(values) for qid, values in by_question.items()}
    return dict(
        questions=len(rates),
        samples=len(rows),
        pass_at_1=mean(rates.values()),
        by_question=rates,
        generated_tokens=sum(len(r["generated_ids"]) for r in rows),
        cap_fraction=mean(r["hit_token_cap"] for r in rows),
        grading_counts=dict(Counter(r["grading_status"] for r in rows)),
    )


def credit_summary(credits):
    tokens = [v for c in credits for v in c["token_advantages"]]
    blocks = [b["advantage"] for c in credits for b in c["blocks"]]
    return dict(
        tokens=len(tokens),
        blocks=len(blocks),
        mean=mean(tokens),
        mean_absolute=mean(map(abs, tokens)),
        mean_square=mean(v * v for v in tokens),
        zero_fraction=mean(v == 0 for v in tokens),
        nonzero_roots=sum(any(v != 0 for v in c["token_advantages"]) for c in credits),
    )


def verify_resume(state, plan_sha, credit, topology, *, plan=None):
    assert state["plan_sha256"] == plan_sha, (
        "Resume must use the original scientific plan"
    )
    assert state["credit"] == credit
    assert state["topology"] == topology, (
        "Native sharded checkpoints require the same topology"
    )
    checkpoint_plan_sha = plan_sha
    if "continuation" in state or (plan is not None and "continuation" in plan):
        from rllab.math_mc_cross_risk_continuation import verify_state

        assert plan is not None and "mc_cross_risk" in plan
        checkpoint_plan_sha = verify_state(plan, state)
    if state["checkpoint"] is not None:
        marker = json.loads((Path(state["checkpoint"]) / "complete.json").read_text())
        assert (
            marker["step"] == state["step"]
            and marker["plan_sha256"] == checkpoint_plan_sha
        )
        assert marker["credit"] == credit and marker["topology"] == topology
    else:
        assert state["step"] == 0
    assert [r["step"] for r in state["history"]] == list(range(1, state["step"] + 1))
