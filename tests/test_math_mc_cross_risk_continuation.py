"""Checkpoint ownership, immutable continuation controls and driver recovery."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rllab.math_io import atomic_json, fingerprint
from rllab.math_mc_cross_risk_continuation import (
    historical_evaluation_plan,
    initial_state,
    make_plan,
    validate_continuation,
)
from rllab.math_mc_cross_risk_training import make_plan as make_parent, validate_plan
from rllab.math_training import training_root_jobs, verify_resume


@pytest.fixture
def continuation(tmp_path):
    train = [dict(question_id=str(i), answer="1") for i in range(1024)]
    test = [dict(question_id=f"t{i}", answer="1") for i in range(512)]
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
            model="mock",
            eval_samples=4,
        ),
        train=train,
        evaluation=test,
        eos=[2],
        schedule=[[str(q) for q in range(i * 16, (i + 1) * 16)] for i in range(64)],
        prompts={q["question_id"]: [1] for q in train + test},
        counts={},
    )
    arm = tmp_path / "parent/prm"
    checkpoint = arm / "attempts/0000/checkpoints/step-00024"
    checkpoint.mkdir(parents=True)
    parent = make_parent(original)
    atomic_json(arm / "plan.json", parent)
    parent_sha = fingerprint(arm / "plan.json")
    topology = dict(nodes=1, devices_per_node=16)
    marker = dict(
        step=24,
        plan_sha256=parent_sha,
        credit="prm",
        topology=topology,
        signatures=[24],
    )
    atomic_json(checkpoint / "complete.json", marker)
    for rank in range(16):
        for kind in ("model", "optim", "extra_state"):
            (checkpoint / f"{kind}_world_size_16_rank_{rank}.pt").write_bytes(
                b"fixture"
            )
    state = dict(
        status="completed",
        step=24,
        plan_sha256=parent_sha,
        credit="prm",
        topology=topology,
        checkpoint=str(checkpoint),
        checkpoints=[str(checkpoint)],
        initial_signatures=[0],
        history=[dict(step=i) for i in range(1, 25)],
        evaluations={str(i): dict(step=i) for i in (0, 8, 16, 24)},
    )
    inputs = dict(
        original_plan=original,
        parent_state=state,
        audit=dict(
            status="completed",
            completed_updates=24,
            mc_grades_independently_recomputed=True,
            plan_sha256=parent_sha,
        ),
    )
    refs = {}
    for key, value in inputs.items():
        path = arm / ("state.json" if key == "parent_state" else f"{key}.json")
        atomic_json(path, value)
        refs[key] = dict(path=str(path), sha256=fingerprint(path))
    refs.update(
        parent_plan=dict(path=str(arm / "plan.json"), sha256=parent_sha),
        marker=dict(
            path=str(checkpoint / "complete.json"),
            sha256=fingerprint(checkpoint / "complete.json"),
        ),
    )
    return make_plan(original, parent, refs), original, arm


def test_continuation_preserves_versions_and_parent_ownership(continuation):
    plan, original, arm = continuation
    before = (arm / "state.json").read_bytes()
    validate_plan(plan, "prm", 16, 1)
    state = initial_state(plan, "new-plan", "prm", dict(nodes=1, devices_per_node=16))
    verify_resume(state, "new-plan", "prm", state["topology"], plan=plan)
    assert state["step"] == 24 and state["checkpoints"] == []
    assert (arm / "state.json").read_bytes() == before
    for version in range(64):
        assert training_root_jobs(plan, version) == training_root_jobs(
            original, version
        )
    comparison = historical_evaluation_plan(plan)
    assert comparison["evaluation"] == original["evaluation"]
    assert comparison["study"]["evaluation_seed"] == 20260914
    assert plan["study"]["evaluation_seed"] == 20260922
    with pytest.raises(AssertionError):
        verify_resume(state, "new-plan", "prm", state["topology"])
    with pytest.raises(AssertionError):
        verify_resume(
            state, "new-plan", "prm", dict(nodes=1, devices_per_node=8), plan=plan
        )
    state["checkpoints"] = [state["checkpoint"]]
    with pytest.raises(AssertionError):
        verify_resume(state, "new-plan", "prm", state["topology"], plan=plan)


@pytest.mark.parametrize(
    "field", ["seed", "learning_rate", "target_parameter_step_l2", "evaluation_seed"]
)
def test_continuation_rejects_silent_control_changes(continuation, field):
    plan, _, _ = continuation
    plan["study"][field] *= 2
    with pytest.raises(AssertionError):
        validate_continuation(plan)


def test_continuation_rejects_changed_parent_and_missing_shard(continuation):
    plan, _, arm = continuation
    state = validate_continuation(plan)
    shard = Path(state["checkpoint"]) / "extra_state_world_size_16_rank_15.pt"
    shard.unlink()
    with pytest.raises(AssertionError, match="extra_state"):
        initial_state(plan, "new-plan", "prm", state["topology"])
    (arm / "state.json").write_text("{}")
    with pytest.raises(AssertionError):
        validate_continuation(plan)


@pytest.mark.real_rl
def test_driver_runs_only_remaining40_and_recovers_final_eval(
    continuation, tmp_path, monkeypatch, real_rl_stack
):
    from rllab import math_train as driver
    from rllab import math_mc_cross_risk_training_runtime as selector
    from rllab import real_rl_runtime as memory
    from rllab.real_math_loop import assemble_credits, root_jobs

    plan, original, parent_arm = continuation
    parent_files = {p: p.read_bytes() for p in parent_arm.rglob("*") if p.is_file()}
    plan_path = tmp_path / "plan.json"
    atomic_json(plan_path, plan)
    restores, sampled, updates = [], [], []
    fail_comparison = [True]
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    monkeypatch.setattr(
        driver.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    monkeypatch.setattr(memory, "host_memory_limit", lambda: 400 * 1024**3)
    monkeypatch.setattr(driver.ray, "shutdown", lambda: None)
    monkeypatch.setattr(
        driver,
        "signal",
        SimpleNamespace(
            SIGTERM=15, SIGALRM=14, signal=lambda *a: None, alarm=lambda *a: None
        ),
    )
    monkeypatch.setattr(driver, "policy_batch", lambda *a, **k: None)
    monkeypatch.setattr(driver, "tu", SimpleNamespace(get=lambda obj, key: obj[key]))

    class Sampler:
        total_tokens = 0

        def sample(self, client, jobs, *, version, phase, **kw):
            if phase == "eval-historical-v64" and fail_comparison[0]:
                fail_comparison[0] = False
                raise RuntimeError("simulated final comparison interruption")
            sampled.append((version, phase, deepcopy(jobs)))
            return [
                dict(
                    j,
                    generated_ids=[3, 2],
                    log_probs=[-0.5, -0.5],
                    reward=j["root_index"] % 2,
                    hit_token_cap=False,
                    grading_status="ok",
                )
                for j in jobs
            ]

    monkeypatch.setattr(driver, "NativeMathSampler", lambda *a: Sampler())
    monkeypatch.setattr(
        selector,
        "sample_mixed_credit",
        lambda wg, cp, sampler, client, p, roots, v, *a: (
            assemble_credits(roots, [], 32),
            [f"v{v}-suffix-pilot", f"v{v}-suffix-rest"],
            None,
        ),
    )

    def create(cfg, event, *, restore, version, **kw):
        restores.append((restore, version))
        counter = [version]
        assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type == "constant"
        assert cfg.actor_rollout_ref.actor.optim.lr_warmup_steps == 0
        assert cfg.actor_rollout_ref.actor.optim.total_training_steps == 64

        class Worker:
            def checkpoint_signature(self):
                return counter.copy()

            def audit_parameters(self, *, before, **kw):
                if before:
                    assert kw["target_parameter_step_l2"] == 0.01
                    return
                return [
                    dict(
                        optimizer_steps=1,
                        sample_finite=True,
                        optimizer_state_entries=0,
                        frozen_local_numel=0,
                        applied_lr=[0.01],
                        step_scale=dict(global_grad_norm=1.0, nominal_step_l2=0.01),
                    )
                ]

            def update_actor(self, data):
                counter[0] += 1
                updates.append(counter[0])
                return dict(metrics=dict(loss=[0.0], grad_norm=[1.0], lr=[0.0001]))

            def save_checkpoint(self, path, **kw):
                Path(path).mkdir(parents=True)
                (Path(path) / "simulated-shard.pt").write_bytes(b"child")

        return (
            Worker(),
            None,
            SimpleNamespace(
                sleep_replicas=lambda: None, update_weights=lambda **kw: None
            ),
        )

    monkeypatch.setattr(driver, "create_runtime", create)
    args = SimpleNamespace(
        plan=plan_path,
        credit="prm",
        run_dir=tmp_path / "child",
        devices=16,
        nodes=1,
        resume=False,
        check_config=False,
        stop_after=None,
        time_limit=None,
        resume_max_wall_seconds=None,
        rollout_max_num_seqs=64,
        rollout_gpu_memory_utilization=0.4,
    )
    monkeypatch.setattr(driver, "PROCESS_STARTED", driver.time.monotonic())
    with pytest.raises(RuntimeError, match="simulated final"):
        driver.train(args)
    arm = args.run_dir / "prm"
    state = json.loads((arm / "state.json").read_text())
    assert state["step"] == 64 and state["status"] == "running"
    assert "64" in state["evaluations"] and "historical_evaluations" not in state
    assert updates == list(range(25, 65))
    args.resume = True
    monkeypatch.setattr(driver, "PROCESS_STARTED", driver.time.monotonic())
    driver.train(args)
    state = json.loads((arm / "state.json").read_text())
    assert state["status"] == "completed" and state["step"] == 64
    assert [v for _, v in restores] == [24, 64]
    assert updates == list(range(25, 65))
    assert set(state["evaluations"]) == {str(i) for i in range(0, 65, 8)}
    assert set(state["historical_evaluations"]) == {"64"}
    assert [Path(p).name for p in state["checkpoints"]] == ["step-00060", "step-00064"]
    assert not (
        arm / "attempts/0000/checkpoints/step-00028/simulated-shard.pt"
    ).exists()
    assert {p: p.read_bytes() for p in parent_files} == parent_files
    roots = [(v, jobs) for v, phase, jobs in sampled if phase.endswith("-roots")]
    assert [v for v, _ in roots] == list(range(24, 64))
    assert all(jobs == training_root_jobs(original, v) for v, jobs in roots)
    evaluations = [
        (v, phase, jobs) for v, phase, jobs in sampled if phase.startswith("eval-")
    ]
    assert [v for v, _, _ in evaluations] == [32, 40, 48, 56, 64, 64]
    assert evaluations[-1][2] == root_jobs(
        original["evaluation"],
        plan["prompts"],
        "llama",
        0,
        4,
        seed=20260914,
        role="eval",
    )
