"""Native on-policy ORM/MC-PRM training, fixed evaluation and checkpoint resume."""

# Count native-library imports toward each attempt's wall budget.
# ruff: noqa: E402
import time

PROCESS_STARTED = time.monotonic()

import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import traceback
from types import SimpleNamespace

import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.utils import tensordict_utils as tu

from rllab.math_io import atomic_json, fingerprint, now, provenance
from rllab.math_training import (
    actor_policy,
    completion_summary,
    credit_summary,
    outcome_credits,
    parameter_step_target,
    training_root_jobs,
    verify_resume,
)
from rllab.real_math_loop import (
    NativeMathSampler,
    assemble_credits,
    root_jobs,
    suffix_jobs,
)
from rllab.real_rl_runtime import compose_config, create_runtime, plain, policy_batch


def native_config(
    plan,
    devices,
    nodes,
    output,
    *,
    rollout_max_num_seqs=None,
    rollout_gpu_memory_utilization=None,
):
    s = plan["study"]
    parameter_step_target(s)
    actor, temperature = actor_policy(s)
    batch = s["questions_per_update"] * s["roots_per_question"]
    assert batch % (devices * nodes) == 0 and s["roots_per_question"] > 1
    assert s["eval_every"] % s["checkpoint_every"] == 0
    repo = Path(__file__).resolve().parents[2]
    actor_config = dict(
        ppo_mini_batch_size=batch,
        data_loader_seed=s["seed"],
        optim=dict(lr=s["learning_rate"], total_training_steps=s["updates"]),
    )
    rollout_config = dict(seed=s["seed"], temperature=temperature)
    if rollout_max_num_seqs is not None:
        assert rollout_max_num_seqs > 0, "Rollout sequence capacity must be positive"
        rollout_config["max_num_seqs"] = rollout_max_num_seqs
    if rollout_gpu_memory_utilization is not None:
        assert 0 < rollout_gpu_memory_utilization < 1, "Invalid rollout memory fraction"
        rollout_config["gpu_memory_utilization"] = rollout_gpu_memory_utilization
    if actor == "gemma3":
        # Vision is unused on text. Keep its parameters in the root FSDP group,
        # whose forward runs, rather than in never-unsharded child groups.
        actor_config["fsdp_config"] = dict(
            wrap_policy=dict(transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"])
        )
    return compose_config(
        repo,
        repo / "configs/rl/math_native.yaml",
        dict(
            data=dict(train_batch_size=batch),
            actor_rollout_ref=dict(
                model=dict(path=s["model"]),
                actor=actor_config,
                rollout=rollout_config,
            ),
            trainer=dict(
                experiment_name=s["name"],
                total_training_steps=s["updates"],
                n_gpus_per_node=devices,
                nnodes=nodes,
                default_local_dir=str(output),
            ),
        ),
    )


def sampling_concurrency(cfg):
    """Match client capacity to the number and capacity of native replicas."""
    rollout = cfg.actor_rollout_ref.rollout
    replica_size = (
        rollout.tensor_model_parallel_size
        * rollout.data_parallel_size
        * rollout.pipeline_model_parallel_size
    )
    devices = cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes
    assert devices % replica_size == 0
    return (devices // replica_size) * rollout.max_num_seqs


def train(args):
    plan = json.loads(args.plan.read_text())
    s, plan_sha = plan["study"], fingerprint(args.plan)
    actor, temperature = actor_policy(s)
    worker_options = {}
    if "mc_cross_risk" in plan:
        from rllab.math_mc_cross_risk_training import validate_plan as validate_mc_plan
        from rllab.math_mc_cross_risk_training_runtime import (
            OnlineMCRiskWorker,
            sample_mixed_credit,
        )

        validate_mc_plan(plan, args.credit, args.devices, args.nodes)
        if not args.check_config:
            from rllab.real_rl_runtime import host_memory_limit

            assert host_memory_limit() >= 320 * 1024**3, (
                "Online MC selection requires at least320GiB host memory"
            )
        worker_options["worker_class"] = OnlineMCRiskWorker
    retired = {"cross_task_trust", "branch_training", "mc_allocation", "credit_routing"}
    if retired.intersection(plan):
        raise ValueError(
            "This plan uses a retired experimental method; use its recorded source revision"
        )
    if s.get("training_seed_policy") == "question":
        assert args.credit == "orm", (
            "Question-paired training currently supports ORM only"
        )
        training_root_jobs(plan, 0)  # Reject repeated exposures before any sampling.
    target_step_l2 = parameter_step_target(s)
    wall_limit = s["max_wall_seconds"]
    if args.resume_max_wall_seconds is not None:
        assert args.resume, "A cumulative budget extension requires --resume"
        assert args.resume_max_wall_seconds > wall_limit, (
            "The resume ceiling must exceed the original plan's wall budget"
        )
        wall_limit = args.resume_max_wall_seconds
    topology = dict(nodes=args.nodes, devices_per_node=args.devices)
    cfg = native_config(
        plan,
        args.devices,
        args.nodes,
        args.run_dir,
        rollout_max_num_seqs=args.rollout_max_num_seqs,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
    )
    concurrency = sampling_concurrency(cfg)
    if args.check_config:
        print(
            json.dumps(
                dict(
                    credit=args.credit,
                    topology=topology,
                    counts=plan["counts"],
                    batch=cfg.data.train_batch_size,
                    lr=cfg.actor_rollout_ref.actor.optim.lr,
                    target_parameter_step_l2=target_step_l2,
                    actor=actor,
                    temperature=temperature,
                    sampling_concurrency=concurrency,
                    rollout_max_num_seqs=cfg.actor_rollout_ref.rollout.max_num_seqs,
                    rollout_gpu_memory_utilization=(
                        cfg.actor_rollout_ref.rollout.gpu_memory_utilization
                    ),
                )
            )
        )
        return
    arm = args.run_dir.resolve() / args.credit
    if args.resume:
        state = json.loads((arm / "state.json").read_text())
        verify_resume(state, plan_sha, args.credit, topology, plan=plan)
        assert state["status"] != "completed", "This arm has already completed"
    else:
        arm.mkdir(parents=True, exist_ok=False)
        atomic_json(arm / "plan.json", plan)
        state = dict(
            status="running",
            plan_sha256=plan_sha,
            credit=args.credit,
            topology=topology,
            step=0,
            checkpoint=None,
            checkpoints=[],
            history=[],
            evaluations={},
        )
        if "continuation" in plan:
            from rllab.math_mc_cross_risk_continuation import initial_state

            state = initial_state(plan, plan_sha, args.credit, topology)
            verify_resume(state, plan_sha, args.credit, topology, plan=plan)
            atomic_json(arm / "state.json", state)
    state["status"] = "running"
    previous = sorted(arm.glob("attempts/*/report.json"))
    spent = sum(json.loads(p.read_text()).get("elapsed_seconds", 0) for p in previous)
    remaining = wall_limit - spent
    if args.time_limit:
        remaining = min(remaining, args.time_limit)
    assert remaining > 0, "Contracted wall-time budget exhausted"
    folder = arm / "attempts" / f"{len(list(arm.glob('attempts/*'))):04d}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "resolved-config.yaml").write_text(OmegaConf.to_yaml(cfg))
    report = dict(
        status="running",
        started_at=now(),
        **provenance(),
        credit=args.credit,
        plan_sha256=plan_sha,
        resumed_from=state["checkpoint"],
        resumed_step=state["step"],
        wall_budget=dict(
            plan_max_wall_seconds=s["max_wall_seconds"],
            effective_max_wall_seconds=wall_limit,
            previous_elapsed_seconds=spent,
            remaining_seconds=remaining,
        ),
        elapsed_seconds=0,
        events=[],
    )

    def event(stage, **fields):
        elapsed = time.monotonic() - PROCESS_STARTED
        report["elapsed_seconds"] = elapsed
        row = dict(
            stage=stage, timestamp=now(), elapsed_seconds=elapsed, **plain(fields)
        )
        report["events"].append(row)
        atomic_json(folder / "report.json", report)
        print(json.dumps(row), flush=True)

    def stop(*_):
        raise TimeoutError("Training wall-time budget or termination")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, int(remaining - (time.monotonic() - PROCESS_STARTED))))
    target = min(args.stop_after or s["updates"], s["updates"])
    # A final checkpoint may be committed before its evaluation or completion
    # marker. Restore it to finish that work without another optimizer step.
    assert state["step"] <= target
    try:
        event(
            "start",
            target_step=target,
            remaining_budget=remaining,
            topology=topology,
            sampling_concurrency=concurrency,
            rollout_max_num_seqs=cfg.actor_rollout_ref.rollout.max_num_seqs,
            rollout_gpu_memory_utilization=(
                cfg.actor_rollout_ref.rollout.gpu_memory_utilization
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(s["model"], local_files_only=True)

        def runtime_event(stage, **fields):
            # A failed IPC receiver can leave the sender blocked. Bound time
            # between initialization stages independently of the 24-hour run.
            remaining_wall = remaining - (time.monotonic() - PROCESS_STARTED)
            signal.alarm(max(1, math.ceil(min(900, remaining_wall))))
            event(stage, **fields)

        runtime_event("runtime_initializing")
        wg, client, checkpoint = create_runtime(
            cfg,
            runtime_event,
            restore=state["checkpoint"],
            version=state["step"],
            **worker_options,
        )
        signal.alarm(
            max(1, math.ceil(remaining - (time.monotonic() - PROCESS_STARTED)))
        )
        signatures = wg.checkpoint_signature()
        if state["checkpoint"]:
            expected = json.loads(
                (Path(state["checkpoint"]) / "complete.json").read_text()
            )
            assert signatures == expected["signatures"], (
                "Restored parameter sample does not match checkpoint"
            )
            event("restore_verified", step=state["step"], signatures=signatures)
        else:
            state["initial_signatures"] = signatures
        questions = {q["question_id"]: q for q in plan["train"] + plan["evaluation"]}
        sampler = NativeMathSampler(
            tokenizer, list(questions.values()), temperature, plan["eos"], folder, event
        )

        def evaluate(step, *, historical=False):
            eval_plan = plan
            if historical:
                from rllab.math_mc_cross_risk_continuation import (
                    historical_evaluation_plan,
                )

                eval_plan = historical_evaluation_plan(plan)
            es = eval_plan["study"]
            phase = f"eval-historical-v{step}" if historical else f"eval-v{step}"
            # Fixed evaluation random keys across checkpoints; actual weights retain step.
            jobs = root_jobs(
                eval_plan["evaluation"],
                plan["prompts"],
                actor,
                0,
                es["eval_samples"],
                seed=es.get("evaluation_seed", es["seed"]),
                role="eval",
            )
            rows = sampler.sample(
                client, jobs, version=step, phase=phase, concurrency=concurrency
            )
            result = dict(
                step=step,
                attempt=str(folder),
                phase=phase,
                summary=completion_summary(rows),
            )
            atomic_json(folder / f"{phase}-summary.json", result)
            key = "historical_evaluations" if historical else "evaluations"
            state.setdefault(key, {})[str(step)] = result
            event(
                "evaluated",
                step=step,
                phase=phase,
                pass_at_1=result["summary"]["pass_at_1"],
            )
            atomic_json(arm / "state.json", state)

        if (
            state["step"] % s["eval_every"] == 0 or state["step"] == s["updates"]
        ) and str(state["step"]) not in state["evaluations"]:
            evaluate(state["step"])
        for version in range(state["step"], target):
            iteration = time.monotonic()
            roots = sampler.sample(
                client,
                training_root_jobs(plan, version),
                version=version,
                phase=f"v{version}-roots",
                concurrency=concurrency,
            )
            suffix_phases = None
            mc_risk_receipt = None
            if "mc_cross_risk" in plan:
                pad = tokenizer.pad_token_id
                if pad is None:
                    pad = tokenizer.eos_token_id
                credits, suffix_phases, mc_risk_receipt = sample_mixed_credit(
                    wg,
                    checkpoint,
                    sampler,
                    client,
                    plan,
                    roots,
                    version,
                    folder,
                    pad,
                    concurrency,
                    event,
                )
            elif args.credit == "prm":
                suffixes = sampler.sample(
                    client,
                    suffix_jobs(
                        roots,
                        plan["prompts"],
                        actor,
                        version,
                        s["mc_samples"],
                        seed=s["seed"],
                    ),
                    version=version,
                    phase=f"v{version}-suffixes",
                    concurrency=concurrency,
                )
                credits = assemble_credits(
                    roots,
                    suffixes,
                    s["mc_samples"],
                )
                del suffixes
            else:
                credits = outcome_credits(roots)
            loss_rows = roots
            loss_prompts = [plan["prompts"][r["question_id"]] for r in roots]
            atomic_json(folder / f"v{version}-credits.json", credits)
            checkpoint.sleep_replicas()
            pad_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            data = policy_batch(
                loss_prompts,
                [
                    SimpleNamespace(
                        token_ids=r["generated_ids"], log_probs=r["log_probs"]
                    )
                    for r in loss_rows
                ],
                [c["token_advantages"] for c in credits],
                pad_id,
                temperature=temperature,
            )
            before = time.monotonic()
            applied_target = target_step_l2
            wg.audit_parameters(before=True, target_parameter_step_l2=target_step_l2)
            metrics = plain(tu.get(wg.update_actor(data), "metrics"))
            audits = wg.audit_parameters(before=False)
            update_seconds = time.monotonic() - before
            for key in ("loss", "grad_norm"):
                assert len(metrics[key]) == 1 and math.isfinite(float(metrics[key][0]))
            assert metrics["lr"] == [s["learning_rate"]]
            grad_norm = float(metrics["grad_norm"][0])
            applied_lr = float(audits[0]["applied_lr"][0])
            assert all(
                a["optimizer_steps"] == 1
                and a["sample_finite"]
                and a["optimizer_state_entries"] == 0
                and a["frozen_local_numel"] == 0
                and all(
                    math.isclose(lr, applied_lr, rel_tol=1e-6, abs_tol=0)
                    for lr in a["applied_lr"]
                )
                for a in audits
            )
            if applied_target is None:
                assert applied_lr == s["learning_rate"]
            else:
                assert all(
                    math.isclose(
                        a["step_scale"]["global_grad_norm"], grad_norm, rel_tol=1e-5
                    )
                    and math.isclose(
                        a["step_scale"]["nominal_step_l2"],
                        applied_target,
                        rel_tol=1e-5,
                    )
                    for a in audits
                )
                assert math.isclose(
                    applied_lr * grad_norm, applied_target, rel_tol=1e-5
                )
            step = version + 1
            record = dict(
                step=step,
                attempt=str(folder),
                roots_phase=f"v{version}-roots",
                suffixes_phase=f"v{version}-suffixes" if args.credit == "prm" else None,
                roots=completion_summary(roots),
                credit=credit_summary(credits),
                metrics=metrics,
                update_seconds=update_seconds,
                audit=audits,
                effective_learning_rate=applied_lr,
                ideal_sgd_update_norm=applied_lr * grad_norm,
            )
            if mc_risk_receipt is not None:
                record.update(
                    suffixes_phase=None,
                    suffixes_phases=suffix_phases,
                    mc_cross_risk=mc_risk_receipt,
                )
            event("actor_updated", **record)
            state["history"].append(record)
            state["step"] = step
            saved = step % s["checkpoint_every"] == 0 or step == target
            if saved:
                before = time.monotonic()
                destination = folder / "checkpoints" / f"step-{step:05d}"
                wg.save_checkpoint(str(destination), global_step=step)
                marker = dict(
                    step=step,
                    plan_sha256=plan_sha,
                    credit=args.credit,
                    topology=topology,
                    signatures=wg.checkpoint_signature(),
                )
                atomic_json(destination / "complete.json", marker)
                old_paths = state["checkpoints"]
                state["checkpoints"] = (old_paths + [str(destination)])[-2:]
                state["checkpoint"] = str(destination)
                atomic_json(arm / "state.json", state)
                # Rotate only previously committed complete checkpoints, after publishing the new pointer.
                for old in old_paths:
                    if old not in state["checkpoints"]:
                        assert Path(old).is_relative_to(arm / "attempts")
                        for product in Path(old).iterdir():
                            if product.name == "complete.json":
                                continue
                            if product.is_dir():
                                shutil.rmtree(product)
                            else:
                                product.unlink()
                        event("checkpoint_rotated", path=old)
                record["checkpoint_seconds"] = time.monotonic() - before
                event(
                    "checkpoint_saved",
                    path=str(destination),
                    step=step,
                    seconds=record["checkpoint_seconds"],
                )
            before = time.monotonic()
            checkpoint.update_weights(global_steps=step)
            record["synchronization_seconds"] = time.monotonic() - before
            record["iteration_seconds"] = time.monotonic() - iteration
            event(
                "weights_synchronized",
                version=step,
                seconds=record["synchronization_seconds"],
            )
            if step % s["eval_every"] == 0 or step == s["updates"]:
                evaluate(step)
            if saved:
                atomic_json(arm / "state.json", state)
            del roots, credits, data, loss_rows, loss_prompts
        if (
            ("continuation" in plan or "historical_test_evaluation" in plan)
            and state["step"] == s["updates"]
            and str(state["step"]) not in state.get("historical_evaluations", {})
        ):
            evaluate(state["step"], historical=True)
        state["status"] = "completed" if target == s["updates"] else "paused"
        atomic_json(arm / "state.json", state)
        report["status"] = state["status"]
        report["generated_tokens"] = sampler.total_tokens
    except BaseException as error:
        report.update(
            status="failed", error=dict(type=type(error).__name__, message=str(error))
        )
        (folder / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        ray.shutdown()
        report["finished_at"] = now()
        event("closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--credit", choices=["orm", "prm"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--time-limit", type=int)
    parser.add_argument(
        "--resume-max-wall-seconds",
        type=int,
        help="Explicit recorded cumulative budget extension; requires --resume",
    )
    parser.add_argument("--rollout-max-num-seqs", type=int)
    parser.add_argument("--rollout-gpu-memory-utilization", type=float)
    parser.add_argument("--check-config", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
