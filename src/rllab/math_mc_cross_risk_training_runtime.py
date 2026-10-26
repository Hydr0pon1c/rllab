"""Current-policy MC selection followed by one ordinary native PRM update."""

from copy import deepcopy
import math
import time
from types import SimpleNamespace

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu

from rllab.math_alignment import active_question_batch, aggregate_capture
from rllab.math_alignment_runtime import AlignmentWorker
from rllab.math_io import atomic_json, binding, write_json
from rllab.math_mc_cross_risk import credit_digest, question_contrast, risk
from rllab.math_mc_cross_risk_training import (
    RULE,
    allocation,
    capture_devices,
    pilot_data,
    question_counts,
    reference_batches,
    reference_groups,
    selected_counts,
)
from rllab.real_math_loop import assemble_credits, root_jobs, suffix_jobs
from rllab.real_rl_runtime import plain, policy_batch


class OnlineMCRiskWorker(AlignmentWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_mc_selection(self):
        engine = self.actor.engine
        assert not hasattr(self, "_gradients")
        self._mc_scheduler = deepcopy(engine.lr_scheduler.state_dict())
        self._mc_lrs = [g["lr"] for g in engine.optimizer.param_groups]
        self._mc_rng = torch.get_rng_state()
        self._mc_cuda_rng = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )
        receipt = self.start_capture()
        return dict(**receipt, scheduler_epoch=self._mc_scheduler["last_epoch"])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_mc_selection(self):
        receipt = self.stop_capture()
        engine = self.actor.engine
        engine.lr_scheduler.load_state_dict(deepcopy(self._mc_scheduler))
        for g, lr in zip(engine.optimizer.param_groups, self._mc_lrs, strict=True):
            g["lr"] = lr
        torch.set_rng_state(self._mc_rng)
        if self._mc_cuda_rng is not None:
            torch.cuda.set_rng_state(self._mc_cuda_rng)
        assert engine.lr_scheduler.state_dict() == self._mc_scheduler
        assert torch.equal(torch.get_rng_state(), self._mc_rng)
        if self._mc_cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), self._mc_cuda_rng)
        result = dict(
            **receipt,
            scheduler_epoch=engine.lr_scheduler.last_epoch,
            scheduler_restored=True,
            rng_restored=True,
        )
        del self._mc_scheduler, self._mc_lrs, self._mc_rng, self._mc_cuda_rng
        return result


def select(wg, plan, roots, pilot, references, version, pad, folder, event):
    """No sampling/writes: capture risk and freeze a scalar-only selection receipt."""
    started = time.monotonic()
    devices = capture_devices(plan)
    groups = reference_groups(plan, version)
    saved, pools = pilot_data(roots, pilot)
    counts = question_counts(roots)
    result = dict(
        rule=RULE,
        version=version,
        reference_groups=groups,
        records=[],
        fallback=None,
        scores=dict.fromkeys(counts, 0.0),
    )
    if any(counts.values()):
        result["initial_signatures"] = wg.checkpoint_signature()
        result["begin"] = wg.begin_mc_selection()
        assert all(x["scheduler_epoch"] == version for x in result["begin"])

        def backward(name, rows, credits, *, save=None, accumulate=(), refs=()):
            kept, advantages, scale = active_question_batch(
                rows, credits, multiple=devices
            )
            data = policy_batch(
                [plan["prompts"][r["question_id"]] for r in kept],
                [
                    SimpleNamespace(
                        token_ids=r["generated_ids"],
                        log_probs=(
                            r["log_probs"]
                            if r["log_probs"] is not None
                            else [0.0] * len(r["generated_ids"])
                        ),
                    )
                    for r in kept
                ],
                advantages,
                pad,
                temperature=0.6,
            )
            event("mc_selector_backward_started", version=version, name=name)
            wg.arm_capture(save=save, accumulate=accumulate, references=refs)
            before = time.monotonic()
            metrics = plain(tu.get(wg.update_actor(data), "metrics"))
            receipts = wg.finish_capture()
            captured = aggregate_capture(receipts)
            norm = float(metrics["grad_norm"][0])
            record = dict(
                name=name,
                roots=len(rows),
                computed_roots=len(kept),
                credit_scale=scale,
                credit_digest=credit_digest(rows, credits),
                native_gradient_norm=norm,
                receipts=receipts,
                seconds=time.monotonic() - before,
            )
            result["records"].append(record)
            atomic_json(folder / f"v{version}-mc-cross-risk-pending.json", result)
            assert len(receipts) == devices and math.isclose(
                math.sqrt(captured["norm2"]), norm, rel_tol=1e-4, abs_tol=1e-10
            )
            assert len(result["records"]) <= 81
            event(
                "mc_selector_backward_complete",
                version=version,
                name=name,
                seconds=record["seconds"],
            )
            return captured

        for i, (name, rows, credit) in enumerate(reference_batches(groups, references)):
            backward(f"target_{i}", rows, credit, accumulate=[name])
        g = backward("pilot", roots, saved, save="pilot", refs=["fit0", "fit1"])
        if g["norm2"]:
            for qi, q in enumerate(counts):
                if not counts[q]:
                    continue
                contrasts = [
                    backward(
                        f"q{qi}_z{k}",
                        roots,
                        question_contrast(roots, saved, pools, q, 0, k),
                        refs=["fit0", "fit1", "pilot"],
                    )
                    for k in range(3)
                ]
                result["scores"][q] = risk(g, contrasts, ("fit0", "fit1"))
        else:
            result["fallback"] = "zero_pilot_gradient"
        result["release"] = wg.end_mc_selection()
        assert all(
            x["intercepted_steps"] == len(result["records"])
            and x["scheduler_epoch"] == version
            for x in result["release"]
        )
        result["final_signatures"] = wg.checkpoint_signature()
        assert result["initial_signatures"] == result["final_signatures"]
    else:
        assert not references
        result["fallback"] = "no_prefixes"
    result["allocation"] = allocation(result["scores"], counts)
    result["seconds"] = time.monotonic() - started
    path = folder / f"v{version}-mc-cross-risk.json"
    write_json(path, result)
    event(
        "mc_cross_risk_selection_frozen",
        version=version,
        calls=result["allocation"]["calls"],
        uniform32_calls=32 * sum(counts.values()),
        selector_seconds=result["seconds"],
        fallback=result["fallback"],
    )
    return result, binding(path)


def sample_mixed_credit(
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
):
    s = plan["study"]
    jobs = suffix_jobs(roots, plan["prompts"], s["actor"], version, 32, seed=s["seed"])
    phases = [f"v{version}-suffix-pilot", f"v{version}-suffix-topup"]
    pilot = sampler.sample(
        client,
        [j for j in jobs if j["sample_index"] < 16],
        version=version,
        phase=phases[0],
        concurrency=concurrency,
    )
    groups = reference_groups(plan, version)
    reference_phase = f"v{version}-mc-reference"
    by_id = {q["question_id"]: q for q in plan["train"]}
    questions = [by_id[q] for group in groups for q in group] if jobs else []
    references = sampler.sample(
        client,
        root_jobs(
            questions,
            plan["prompts"],
            s["actor"],
            version,
            2,
            seed=s["seed"],
            role="mc_reference",
        ),
        version=version,
        phase=reference_phase,
        concurrency=concurrency,
    )
    checkpoint.sleep_replicas()
    decision, receipt = select(
        wg, plan, roots, pilot, references, version, pad, folder, event
    )
    # The actor is unchanged. Reuse the verified sync path after capture offload.
    event("mc_selector_weight_sync_start", version=version)
    checkpoint.update_weights(global_steps=version)
    event("mc_selector_weight_sync_done", version=version)
    budgets = decision["allocation"]["budgets"]
    topup = sampler.sample(
        client,
        [
            j
            for j in jobs
            if j["sample_index"] >= 16 and budgets[j["question_id"]] == 32
        ],
        version=version,
        phase=phases[1],
        concurrency=concurrency,
    )
    suffixes = pilot + topup
    assert len(suffixes) == decision["allocation"]["calls"]
    credits = assemble_credits(
        roots,
        suffixes,
        32,
        prefix_counts=selected_counts(roots, decision["allocation"]),
    )
    return (
        credits,
        phases,
        dict(
            **receipt,
            reference_phase=reference_phase,
            reference_calls=len(references),
            mc_calls=len(suffixes),
            uniform32_calls=len(jobs),
            selector_seconds=decision["seconds"],
        ),
    )
