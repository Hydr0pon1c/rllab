"""Scientific pairing and checkpoint-boundary controls for the real-task trainer."""

import copy
import json

import pytest

from rllab.math_io import atomic_json
from rllab.math_training import (
    actor_policy,
    build_plan,
    completion_summary,
    outcome_credits,
    parameter_step_target,
    question_schedule,
    repeat_plan,
    startup_plan,
    validate_repeat,
    verify_resume,
)
from rllab.real_math_loop import root_jobs


def test_question_schedule_and_eval_random_keys():
    questions = [dict(question_id=str(i)) for i in range(11)]
    schedule = question_schedule(questions, 4, 8, 12)
    assert schedule == question_schedule(list(reversed(questions)), 4, 8, 12)
    assert all(len(set(batch)) == 4 for batch in schedule)
    assert len(set(sum(schedule[:2], []))) == 8
    prompts = {q["question_id"]: [5, 7] for q in questions}
    root = root_jobs(questions, prompts, "qwen", 0, 2, seed=12)
    evaluation = root_jobs(questions, prompts, "qwen", 0, 2, seed=12, role="eval")
    assert [r["seed"] for r in root] != [r["seed"] for r in evaluation]
    assert root == root_jobs(questions, prompts, "qwen", 0, 2, seed=12)
    llama = root_jobs(questions, prompts, "llama", 0, 2, seed=12)
    assert [r["seed"] for r in root] != [r["seed"] for r in llama]


def test_actor_policy_preserves_legacy_and_requires_explicit_new_temperature():
    assert actor_policy({}) == ("qwen", 1.0)
    assert actor_policy(dict(actor="llama", temperature=0.6)) == ("llama", 0.6)
    assert actor_policy(dict(actor="gemma3", temperature=1.0)) == ("gemma3", 1.0)
    for settings in (
        dict(actor="llama"),
        dict(actor="gemma3"),
        dict(actor="unknown"),
        dict(temperature=0),
        dict(temperature=float("nan")),
    ):
        with pytest.raises(ValueError):
            actor_policy(settings)


def test_repeat_preserves_population_and_controls_but_changes_training_streams():
    train = [dict(question_id=str(i)) for i in range(8)]
    original = dict(
        study=dict(
            name="fixture",
            seed=12,
            acceptance=False,
            questions_per_update=2,
            updates=4,
            learning_rate=1e-4,
            mc_samples=32,
        ),
        train=train,
        evaluation=[dict(question_id="held-out")],
        prompts={q["question_id"]: [1, 2] for q in train},
        schedule=question_schedule(train, 2, 4, 12),
        counts=dict(
            train=8,
            evaluation=1,
            train_question_slots=8,
            distinct_scheduled_questions=8,
        ),
        source_dataset_sha256="original-dataset",
    )
    before = json.dumps(original, sort_keys=True)
    repeat = repeat_plan(original, 13)
    assert json.dumps(original, sort_keys=True) == before
    for field in ("train", "evaluation", "prompts", "counts", "source_dataset_sha256"):
        assert repeat[field] == original[field]
    assert repeat["schedule"] != original["schedule"]
    assert sorted(sum(repeat["schedule"], [])) == [str(i) for i in range(8)]
    assert repeat["study"] == {
        **original["study"],
        "name": "fixture-s13",
        "seed": 13,
        "evaluation_seed": 12,
    }
    repeat["train"][0]["question_id"] = "changed"
    assert json.dumps(original, sort_keys=True) == before
    for invalid in (12, -1, True):
        with pytest.raises(ValueError):
            repeat_plan(original, invalid)


def test_repeat_rejects_changed_questions_and_policy():
    parent = dict(
        train=["q"],
        evaluation=["t"],
        prompts={"q": [1]},
        eos=[2],
        native_model_revision="fixed",
        source_dataset_sha256="data",
        study=dict(seed=20260914, name="parent", learning_rate=1e-4),
    )
    plan = copy.deepcopy(parent)
    plan["study"].update(seed=1001, name="parent-s1001", evaluation_seed=20260914)
    validate_repeat(parent, plan)
    for field in ("train", "prompts", "native_model_revision"):
        changed = copy.deepcopy(plan)
        changed[field] = None
        with pytest.raises(AssertionError):
            validate_repeat(parent, changed)
    plan["study"]["learning_rate"] *= 2
    with pytest.raises(AssertionError):
        validate_repeat(parent, plan)


@pytest.fixture(scope="module")
def math_analyzer():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/analyze_math_training.py"
    spec = importlib.util.spec_from_file_location("math_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "role,evaluation_seed,request_seed,seed_policy",
    [
        ("root", None, 12, "version"),
        ("eval", None, 12, "version"),
        ("root", 99, 12, "version"),
        ("eval", 99, 99, "version"),
        ("root", 99, 12, "question"),
        ("eval", 99, 99, "question"),
    ],
)
def test_independent_reconstruction_uses_separate_eval_seed(
    tmp_path, math_analyzer, role, evaluation_seed, request_seed, seed_policy
):
    from rllab.math_credit import keyed_seed

    study = dict(seed=12)
    if evaluation_seed is not None:
        study["evaluation_seed"] = evaluation_seed
    plan = dict(study=study, train=[dict(question_id="q")], evaluation=[], eos=[2])
    if seed_policy == "question":
        study["training_seed_policy"] = seed_policy
        plan["schedule"] = [["q"]]
    grade = dict(
        root_slot=0, position=0, sample_index=0, reward=1, grading_status="correct"
    )
    row = dict(
        root_slot=0,
        position=0,
        sample_index=0,
        root_index=0,
        question_id="q",
        role=role,
        weight_version=16,
        extra_fields=dict(global_steps=16, min_global_steps=16, max_global_steps=16),
        generated_ids=[1, 2],
        remaining_budget=3072,
        hit_token_cap=False,
        log_probs=[-0.4, -0.2] if role == "root" else None,
        seed=keyed_seed(
            request_seed,
            "native-math",
            "qwen",
            0 if role == "eval" or seed_policy == "question" else 16,
            "q",
            role,
            0,
        ),
    )
    (tmp_path / f"{role}.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / f"{role}-grades.jsonl").write_text(json.dumps(grade) + "\n")
    result = math_analyzer.reconstruct_phase(
        tmp_path, role, plan, 16, None, regrade=False
    )
    assert result[0]["reward"] == 1
    row["seed"] += 1
    (tmp_path / f"{role}.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError):
        math_analyzer.reconstruct_phase(tmp_path, role, plan, 16, None, regrade=False)


@pytest.mark.parametrize("actor,temperature", [("llama", 0.6), ("gemma3", 1.0)])
def test_dapo_reference_pairing_and_train_only_startup(tmp_path, actor, temperature):
    class Tokenizer:
        blocked = None

        def apply_chat_template(self, messages, **kwargs):
            return [1] * (
                1025 if messages[0]["content"].endswith(str(self.blocked)) else 7
            )

    rows = [
        dict(
            question_id=f"{split}-{i}",
            group_id=f"{split}-{i}",
            split=split,
            source="dapo",
            problem=f"problem {split}-{i}",
            answer=str(i),
        )
        for split in ("train", "test")
        for i in range(6)
    ]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("".join(json.dumps(q) + "\n" for q in rows))
    (tmp_path / "generation_config.json").write_text(
        json.dumps(dict(eos_token_id=[2, 3]))
    )
    study = dict(
        dataset=str(dataset),
        model=str(tmp_path),
        source="dapo",
        level=None,
        seed=12,
        train_questions=4,
        test_questions=3,
        questions_per_update=2,
        roots_per_question=8,
        updates=2,
        acceptance=False,
        name="fixture",
    )
    qwen = build_plan(study, Tokenizer())
    tokenizer = Tokenizer()
    llama_study = {**study, "actor": actor, "temperature": temperature}
    llama = build_plan(llama_study, tokenizer, reference=qwen)
    assert all(llama[k] == qwen[k] for k in ("train", "evaluation", "schedule"))
    assert llama["native_model_revision"] != qwen["native_model_revision"]
    assert all(q["source"] == "dapo" for q in llama["train"])
    probe = startup_plan(llama)
    assert probe["study"]["acceptance"] and probe["study"]["updates"] == 1
    assert probe["study"]["train_questions"] == probe["counts"]["train"] == 2
    assert probe["train"] == probe["evaluation"]
    assert [q["question_id"] for q in probe["train"]] == llama["schedule"][0]
    assert not set(probe["prompts"]) & {q["question_id"] for q in llama["evaluation"]}
    assert llama["study"]["updates"] == 2  # Derivation must not mutate the formal plan.
    tokenizer.blocked = qwen["evaluation"][0]["question_id"]
    with pytest.raises(AssertionError, match="Reference test questions"):
        build_plan(llama_study, tokenizer, reference=qwen)
    with pytest.raises(ValueError):
        build_plan({**study, "level": 5}, Tokenizer())


def test_orm_uses_same_question_leave_one_out_and_token_sum():
    roots = [
        dict(question_id=q, root_index=i, reward=r, generated_ids=[7] * n)
        for q, i, r, n in [
            ("q", 0, 1, 3),
            ("q", 1, 0, 1),
            ("q", 2, 0, 2),
            ("z", 0, 1, 4),
            ("z", 1, 1, 2),
        ]
    ]
    c = outcome_credits(roots)
    assert [x["token_advantages"] for x in c] == [
        [1.0] * 3,
        [-0.5],
        [-0.5] * 2,
        [0.0] * 4,
        [0.0] * 2,
    ]
    for row, credit in zip(roots, c):
        assert len(credit["token_advantages"]) == len(row["generated_ids"])


def test_resume_rejects_wrong_arm_plan_topology_and_uncommitted_steps(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    topology = dict(nodes=1, devices_per_node=2)
    marker = dict(step=8, plan_sha256="frozen", credit="prm", topology=topology)
    atomic_json(checkpoint / "complete.json", marker)
    state = dict(
        **marker,
        checkpoint=str(checkpoint),
        history=[dict(step=i) for i in range(1, 9)],
    )
    verify_resume(state, "frozen", "prm", topology)
    for plan, arm, world in [
        ("changed", "prm", topology),
        ("frozen", "orm", topology),
        ("frozen", "prm", dict(nodes=1, devices_per_node=4)),
    ]:
        with pytest.raises(AssertionError):
            verify_resume(state, plan, arm, world)
    with pytest.raises(AssertionError):
        verify_resume({**state, "step": 9}, "frozen", "prm", topology)
    atomic_json(tmp_path / "state.json", state)
    assert json.loads((tmp_path / "state.json").read_text()) == state
    assert not (tmp_path / "state.json.tmp").exists()


def test_evaluation_reports_sampled_pass_at_one_not_any_success():
    rows = [
        dict(
            question_id=q,
            reward=r,
            generated_ids=[1],
            hit_token_cap=False,
            grading_status="correct" if r else "extracted_wrong",
        )
        for q, r in [("a", 1), ("a", 0), ("b", 0), ("b", 0)]
    ]
    summary = completion_summary(rows)
    assert summary["pass_at_1"] == 0.25
    assert summary["by_question"] == {"a": 0.5, "b": 0}


def test_parameter_step_rule_rejects_ambiguous_or_invalid_plans():
    assert parameter_step_target({}) is None
    assert parameter_step_target(dict(scale_rule="fixed_learning_rate")) is None
    assert (
        parameter_step_target(
            dict(scale_rule="fixed_parameter_step", target_parameter_step_l2=0.01)
        )
        == 0.01
    )
    for value in (None, 0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            parameter_step_target(
                dict(scale_rule="fixed_parameter_step", target_parameter_step_l2=value)
            )
    with pytest.raises(ValueError):
        parameter_step_target(dict(target_parameter_step_l2=0.01))


@pytest.mark.real_rl
def test_sgd_fixed_step_uses_accumulated_gradient_and_preserves_direction(
    real_rl_stack,
):
    import torch
    from rllab.real_rl_runtime import apply_parameter_step

    for multiplier in (1.0, 7.0):
        p = torch.nn.Parameter(torch.zeros(2))
        q = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([{"params": [p]}, {"params": [q]}], lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
        records = []
        hook = opt.register_step_pre_hook(
            lambda optimizer, args, kwargs: records.append(
                apply_parameter_step(optimizer, 0.01)
            )
        )
        # Two microbatches with different directions: normalize only their sum.
        (multiplier * (p[0] + 2 * q[0])).backward()
        (multiplier * (2 * p[0] + 4 * p[1] - 2 * q[0])).backward()
        original = torch.cat([p.grad, q.grad]).clone()
        opt.step()
        scheduler.step()
        assert torch.allclose(torch.cat([p, q]), torch.tensor([-0.006, -0.008, 0.0]))
        assert torch.equal(torch.cat([p.grad, q.grad]), original)
        assert records[0]["global_grad_norm"] == pytest.approx(5 * multiplier)
        assert records[0]["effective_lr"] == pytest.approx(0.002 / multiplier)
        assert records[0]["nominal_step_l2"] == pytest.approx(0.01)
        assert scheduler.get_last_lr() == [1e-4, 1e-4]
        hook.remove()
    for value in (0.0, float("nan"), float("inf")):
        p.grad.fill_(value)
        q.grad.fill_(value)
        before = p.detach().clone()
        with pytest.raises((ValueError, RuntimeError)):
            apply_parameter_step(opt, 0.01)
        assert torch.equal(p, before)


def _check_sharded_step(rank, rendezvous):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor
    from rllab.real_rl_runtime import apply_parameter_step

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        grads = [torch.arange(1.0, 6.0), torch.tensor([6.0, 8.0])]
        parameters = []
        for grad, placement in zip(grads, (Shard(0), Replicate())):
            p = torch.nn.Parameter(
                distribute_tensor(torch.zeros_like(grad), mesh, [placement])
            )
            p.grad = distribute_tensor(grad, mesh, [placement])
            parameters.append(p)
        opt = torch.optim.SGD(parameters, lr=1e-4)
        expected_norm = torch.cat(grads).norm().item()
        measured = apply_parameter_step(opt, 0.01)
        assert measured["global_grad_norm"] == pytest.approx(expected_norm)
        opt.step()
        for p, g in zip(parameters, grads):
            assert torch.allclose(p.full_tensor(), -0.01 * g / expected_norm)
    finally:
        dist.destroy_process_group()


@pytest.mark.real_rl
def test_fixed_step_counts_uneven_shards_and_replicated_parameters_once(
    tmp_path, real_rl_stack
):
    import torch.multiprocessing as mp

    mp.spawn(_check_sharded_step, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


@pytest.mark.real_rl
def test_worker_records_applied_lr_and_bf16_displacement(real_rl_stack):
    from types import SimpleNamespace
    import torch
    from rllab.real_rl_runtime import AuditedActorWorker

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(1.0)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    engine = SimpleNamespace(module=model, optimizer=opt)
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine), rank=0)
    audit = AuditedActorWorker.audit_parameters
    for target in (None, 0.01):
        before = model.weight.detach().clone()
        audit(worker, before=True, target_parameter_step_l2=target)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        opt.step()
        # The native scheduler returns its reference LR after the optimizer call.
        opt.param_groups[0]["lr"] = 1e-4
        result = audit(worker, before=False)
        after = model.weight.detach()
        assert result["optimizer_steps"] == 1
        assert result["applied_lr"] == [1e-4 if target is None else 0.002]
        assert result["sampled_bf16_update_norm"] == pytest.approx(
            (after.bfloat16().float() - before.bfloat16().float()).norm().item()
        )
        assert result["sampled_update_norm"] == pytest.approx(
            (after - before).norm().item()
        )
        assert not opt._optimizer_step_pre_hooks and not opt._optimizer_step_post_hooks


@pytest.mark.real_rl
@pytest.mark.parametrize(
    "rollout_options,sequences,memory",
    [
        ({}, 64, 0.4),
        (
            dict(rollout_max_num_seqs=128, rollout_gpu_memory_utilization=0.6),
            128,
            0.6,
        ),
    ],
)
def test_native_execution_topology_keeps_scientific_batch_and_optimizer(
    tmp_path, real_rl_stack, rollout_options, sequences, memory
):
    from pathlib import Path
    import yaml
    from rllab.math_train import native_config, sampling_concurrency

    repo = Path(__file__).resolve().parents[1]
    study = yaml.safe_load((repo / "configs/rl/qwen_math.yaml").read_text())["study"]
    for devices, actor, temperature in (
        (2, "qwen", 1.0),
        (8, "qwen", 1.0),
        (16, "qwen", 1.0),
        (2, "llama", 0.6),
        (8, "llama", 0.6),
        (16, "llama", 0.6),
        (4, "gemma3", 1.0),
        (8, "gemma3", 1.0),
    ):
        cfg = native_config(
            dict(study={**study, "actor": actor, "temperature": temperature}),
            devices,
            1,
            tmp_path,
            **rollout_options,
        )
        assert (
            cfg.data.train_batch_size
            == cfg.actor_rollout_ref.actor.ppo_mini_batch_size
            == 128
        )
        assert cfg.actor_rollout_ref.actor.optim.lr == 1e-4
        assert parameter_step_target(study) == 0.01
        assert cfg.actor_rollout_ref.actor.fsdp_config.model_dtype == "fp32"
        assert cfg.actor_rollout_ref.actor.fsdp_config.dtype == "bfloat16"
        wrap_classes = cfg.actor_rollout_ref.actor.fsdp_config.wrap_policy.get(
            "transformer_layer_cls_to_wrap"
        )
        assert wrap_classes == (["Gemma3DecoderLayer"] if actor == "gemma3" else None)
        assert cfg.actor_rollout_ref.rollout.dtype == "bfloat16"
        assert cfg.actor_rollout_ref.rollout.temperature == temperature
        assert cfg.actor_rollout_ref.actor.optim.optimizer == "SGD"
        assert cfg.actor_rollout_ref.actor.optim.override_optimizer_config.momentum == 0
        assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "bypass_mode"
        assert cfg.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum"
        assert cfg.trainer.n_gpus_per_node == devices
        assert cfg.actor_rollout_ref.rollout.max_num_seqs == sequences
        assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == memory
        assert cfg.actor_rollout_ref.rollout.max_num_batched_tokens == 4096
        assert sampling_concurrency(cfg) == sequences * devices
        cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
        assert sampling_concurrency(cfg) == sequences * devices // 2


@pytest.mark.real_rl
@pytest.mark.parametrize("extra_evaluation", [False, True])
@pytest.mark.parametrize(
    "resume_limit,previous_elapsed",
    [(None, (10, 20)), (7200, (1800, 1801)), (None, (1800, 1801))],
)
def test_resume_final_checkpoint_finishes_evaluation_without_update(
    tmp_path,
    monkeypatch,
    real_rl_stack,
    resume_limit,
    previous_elapsed,
    extra_evaluation,
):
    from pathlib import Path
    from types import SimpleNamespace
    import yaml
    from rllab import math_train as driver
    from rllab.math_io import fingerprint

    repo = Path(__file__).resolve().parents[1]
    study = yaml.safe_load((repo / "configs/rl/qwen_math.yaml").read_text())["study"]
    study["max_wall_seconds"] = 3600
    plan_path = tmp_path / "plan.json"
    plan = dict(study=study, train=[], evaluation=[], prompts={}, eos=[])
    if extra_evaluation:
        plan["evaluation"] = [dict(question_id="q")]
        plan["prompts"] = {"q": [1]}
        plan["historical_test_evaluation"] = dict(
            step=64, seed=99, samples=4, question_ids=["q"]
        )
    atomic_json(plan_path, plan)
    arm = tmp_path / "orm"
    checkpoint = arm / "attempts/0000/checkpoints/step-00064"
    checkpoint.mkdir(parents=True)
    for index, elapsed in enumerate(previous_elapsed):
        (arm / f"attempts/{index:04d}").mkdir(exist_ok=True)
        atomic_json(
            arm / f"attempts/{index:04d}/report.json",
            dict(status="failed", elapsed_seconds=elapsed),
        )
    record = dict(
        step=64,
        plan_sha256=fingerprint(plan_path),
        credit="orm",
        topology=dict(nodes=1, devices_per_node=8),
    )
    atomic_json(checkpoint / "complete.json", dict(**record, signatures=[]))
    atomic_json(
        arm / "state.json",
        dict(
            **record,
            status="running",
            checkpoint=str(checkpoint),
            checkpoints=[str(checkpoint)],
            history=[dict(step=i) for i in range(1, 65)],
            evaluations={},
        ),
    )
    calls = []

    def restore(cfg, event, *, restore, version):
        assert restore == str(checkpoint) and version == 64
        # No update_actor method: any accidental extra update fails this test.
        return SimpleNamespace(checkpoint_signature=lambda: []), None, None

    def sample(client, jobs, **kwargs):
        calls.append(kwargs["version"])
        return []

    monkeypatch.setattr(driver, "create_runtime", restore)
    monkeypatch.setattr(driver.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(
        driver,
        "NativeMathSampler",
        lambda *a: SimpleNamespace(sample=sample, total_tokens=0),
    )
    monkeypatch.setattr(driver, "completion_summary", lambda rows: dict(pass_at_1=0))
    monkeypatch.setattr(driver.ray, "shutdown", lambda: None)
    monkeypatch.setattr(
        driver,
        "signal",
        SimpleNamespace(
            SIGTERM=15, SIGALRM=14, signal=lambda *a: None, alarm=lambda *a: None
        ),
    )
    monkeypatch.setattr(driver, "PROCESS_STARTED", driver.time.monotonic())
    args = SimpleNamespace(
        plan=plan_path,
        credit="orm",
        run_dir=tmp_path,
        devices=8,
        nodes=1,
        resume=True,
        check_config=False,
        time_limit=None,
        resume_max_wall_seconds=resume_limit,
        stop_after=None,
        rollout_max_num_seqs=None,
        rollout_gpu_memory_utilization=None,
    )
    original_plan = plan_path.read_bytes()
    if resume_limit is None and sum(previous_elapsed) >= 3600:
        with pytest.raises(AssertionError, match="wall-time budget exhausted"):
            driver.train(args)
        assert not (arm / "attempts/0002").exists()
        return
    driver.train(args)
    state = json.loads((arm / "state.json").read_text())
    assert calls == ([64, 64] if extra_evaluation else [64])
    assert state["status"] == "completed" and state["step"] == 64
    assert len(state["history"]) == 64 and "64" in state["evaluations"]
    if extra_evaluation:
        assert state["historical_evaluations"]["64"]["phase"] == "eval-historical-v64"
    assert plan_path.read_bytes() == original_plan
    report = json.loads((arm / "attempts/0002/report.json").read_text())
    assert report["wall_budget"] == dict(
        plan_max_wall_seconds=3600,
        effective_max_wall_seconds=resume_limit or 3600,
        previous_elapsed_seconds=sum(previous_elapsed),
        remaining_seconds=(resume_limit or 3600) - sum(previous_elapsed),
    )


def test_paired_endpoint_and_normalized_auc_are_question_weighted(math_analyzer):
    def arm(credit, values):
        return dict(
            credit=credit,
            plan_sha256="same",
            initial_signatures=["same"],
            status="completed",
            study=dict(updates=64, seed=12, acceptance=False),
            curves={
                str(step): dict(by_question=dict(zip(("a", "b"), row)))
                for step, row in zip((0, 32, 64), values)
            },
        )

    orm = arm("orm", [(0, 0), (0.5, 0), (1, 0)])
    prm = arm("prm", [(0, 0), (0.5, 0.5), (1, 1)])
    result = math_analyzer.compare(orm, prm)
    assert result["prm_minus_orm"]["endpoint"]["estimate"] == 0.5
    assert result["prm_minus_orm"]["improvement"]["estimate"] == 0.5
    assert result["prm_minus_orm"]["normalized_auc"]["estimate"] == 0.25
    assert result["per_arm"]["orm"]["normalized_auc"] == 0.25
    assert result["per_arm"]["prm"]["normalized_auc"] == 0.5
    assert result["prm_minus_orm"]["endpoint"]["question_bootstrap_95"] == [0.0, 1.0]
