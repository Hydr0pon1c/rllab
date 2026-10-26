"""Native capture state restoration and complete online pilot/top-up handoff."""

import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.real_rl


@pytest.mark.parametrize("devices", [8, 16])
def test_online_selection_restores_then_trains_once(
    real_rl_stack, monkeypatch, tmp_path, devices
):
    import torch
    from rllab import math_mc_cross_risk_training_runtime as runtime
    from rllab.math_mc_cross_risk_training import SPEC, audit_selection

    model = torch.nn.Linear(12, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = SimpleNamespace(
        module=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=None,
        optimizer_config=SimpleNamespace(clip_grad=float("inf")),
    )
    worker = SimpleNamespace(rank=0, actor=SimpleNamespace(engine=engine))
    for name in (
        "begin_mc_selection",
        "end_mc_selection",
        "start_capture",
        "stop_capture",
        "arm_capture",
        "finish_capture",
        "checkpoint_signature",
    ):
        setattr(
            worker, name, MethodType(getattr(runtime.OnlineMCRiskWorker, name), worker)
        )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: devices)
    monkeypatch.setattr(runtime.tu, "get", lambda data, key: data[key])
    rng = np.random.default_rng(713)
    features = torch.tensor(rng.normal(size=(35000, 12)), dtype=torch.float32)

    def policy(prompts, outputs, credits, pad, temperature):
        assert temperature == 0.6
        assert all(len(o.log_probs) == len(o.token_ids) for o in outputs)
        return sum(
            (features[o.token_ids] * torch.tensor(c)[:, None]).sum(0)
            for o, c in zip(outputs, credits, strict=True)
        ) / len(outputs)

    monkeypatch.setattr(runtime, "policy_batch", policy)

    class Group:
        def __getattr__(self, name):
            def call(*a, **kw):
                r = getattr(worker, name)(*a, **kw)
                if r is None:
                    return None
                return [dict(r, rank=i) for i in range(devices)]

            return call

        def update_actor(self, gradient):
            optimizer.zero_grad()
            (model.weight * gradient).sum().backward()
            norm = float(model.weight.grad.norm())
            optimizer.step()
            scheduler.step()
            torch.rand(2)  # Native diagnostic work must not consume training RNG.
            return dict(metrics=dict(grad_norm=[norm]))

    group = Group()
    plan = dict(
        study=dict(actor="llama", seed=20260914),
        train=[dict(question_id=str(i)) for i in range(1024)],
        schedule=[[str(i) for i in range(16)]] * 2,
        prompts={str(i): [1] for i in range(1024)},
        mc_cross_risk=dict(SPEC, devices=devices),
    )
    roots = [
        dict(
            question_id=str(q),
            root_index=i,
            root_slot=q * 8 + i,
            reward=i % 2,
            generated_ids=[q * 32 + i + j for j in range(33)],
            log_probs=[-0.5] * 33,
        )
        for q in range(16)
        for i in range(8)
    ]
    # Pruned contrasts must preserve the original 128-root mean loss, including
    # the zero-credit question needed to fill sixteen ranks for eight real roots.
    prune = runtime.active_question_batch

    def checked_prune(rows, credits, *, multiple):
        kept, advantages, scale = prune(rows, credits, multiple=multiple)

        def gradient(batch, values):
            return policy(
                [],
                [
                    SimpleNamespace(
                        token_ids=r["generated_ids"],
                        log_probs=[0.0] * len(r["generated_ids"]),
                    )
                    for r in batch
                ],
                values,
                0,
                0.6,
            )

        torch.testing.assert_close(
            gradient(kept, advantages),
            gradient(rows, [c["token_advantages"] for c in credits]),
        )
        assert len(kept) % devices == 0
        active = {
            r["question_id"]
            for r, c in zip(rows, credits, strict=True)
            if any(c["token_advantages"])
        }
        if len(rows) == 128 and len(active) == 1:
            assert len(kept) == devices and scale == devices / 128
        return kept, advantages, scale

    monkeypatch.setattr(runtime, "active_question_batch", checked_prune)
    phases, events, lifecycle = {}, [], []

    def event(stage, **fields):
        events.append(dict(stage=stage, **fields))

    class Sampler:
        def sample(self, client, jobs, *, version, phase, concurrency):
            assert lifecycle[-1] == "awake"
            event("sampling_started", phase=phase, version=version)
            rows = []
            for job in jobs:
                q = int(job["question_id"])
                ids = [q * 32 + job["root_index"] + j for j in range(33)]
                reward = (
                    job["root_index"] % 2
                    if job["role"] == "mc_reference"
                    else int(
                        (job["sample_index"] * 7 + q + job["root_index"]) % 11
                        < q % 8 + 1
                    )
                )
                rows.append(dict(job, generated_ids=ids, log_probs=None, reward=reward))
            phases[phase] = rows
            return rows

    class Checkpoint:
        def sleep_replicas(self):
            lifecycle.append("sleep")

        def update_weights(self, global_steps):
            assert not hasattr(worker, "_gradients")
            assert scheduler.last_epoch == global_steps
            lifecycle.append("awake")

    for version in range(2):
        lifecycle.append("awake")
        initial = model.weight.detach().clone()
        state = scheduler.state_dict().copy()
        random_state = torch.get_rng_state().clone()
        credits, names, ref = runtime.sample_mixed_credit(
            group,
            Checkpoint(),
            Sampler(),
            None,
            plan,
            roots,
            version,
            tmp_path,
            0,
            8,
            event,
        )
        assert torch.equal(model.weight, initial)
        assert scheduler.state_dict() == state
        assert torch.equal(torch.get_rng_state(), random_state)
        assert not hasattr(worker, "_gradients") and not optimizer.state
        assert lifecycle[-3:] == ["awake", "sleep", "awake"]
        receipt = json.loads((tmp_path / f"v{version}-mc-cross-risk.json").read_text())
        assert len(receipt["records"]) == 81
        contrasts = [r for r in receipt["records"] if r["name"].startswith("q")]
        assert any(r["computed_roots"] == devices for r in contrasts)
        # A fully zero contrast deliberately retains all rows in the shared helper.
        assert all(
            r["computed_roots"] == devices
            or (r["computed_roots"] == 128 and r["native_gradient_norm"] == 0)
            for r in contrasts
        )
        assert ref["mc_calls"] <= ref["uniform32_calls"] * 0.75
        record = dict(
            step=version + 1,
            attempt=str(tmp_path),
            suffixes_phase=None,
            suffixes_phases=names,
            mc_cross_risk=ref,
        )
        (tmp_path / "report.json").write_text(json.dumps(dict(events=events)))
        audit_selection(
            plan,
            roots,
            sum((phases[n] for n in names), []),
            phases[ref["reference_phase"]],
            record,
        )
        full = policy(
            [],
            [
                SimpleNamespace(token_ids=r["generated_ids"], log_probs=r["log_probs"])
                for r in roots
            ],
            [c["token_advantages"] for c in credits],
            0,
            0.6,
        )
        group.update_actor(full)
        assert scheduler.last_epoch == version + 1
        torch.testing.assert_close(model.weight, initial - 0.0001 * full[None, :])
        assert not torch.equal(model.weight, initial)
    # Receipt mutation is rejected before any claimed scientific result.
    changed = json.loads((tmp_path / "v1-mc-cross-risk.json").read_text())
    changed["scores"][roots[0]["question_id"]] += 1
    (tmp_path / "v1-mc-cross-risk.json").write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit_selection(
            plan,
            roots,
            sum((phases[n] for n in names), []),
            phases[ref["reference_phase"]],
            record,
        )
