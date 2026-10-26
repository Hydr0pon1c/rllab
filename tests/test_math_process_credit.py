"""Claim-bearing terminal/block credit and native packing controls."""

from types import SimpleNamespace

import pytest

from rllab.math_credit import process_token_credit


def test_all_root_mc_pools_and_leave_one_out_baselines():
    from rllab.real_math_loop import assemble_credits, suffix_jobs

    roots = [
        dict(question_id="q", root_index=i, generated_ids=[8] * 65, reward=1 - i)
        for i in range(2)
    ]
    jobs = suffix_jobs(roots, {"q": [3, 4]}, "fixture", 0, 2)
    assert len(jobs) == 8
    assert all(j["prompt_ids"] == [3, 4] + [8] * j["position"] for j in jobs)
    for j in jobs:
        j["reward"] = (
            int(j["root_slot"] == 0) if j["position"] == 32 else j["sample_index"]
        )
    credits = assemble_credits(roots, jobs, 2)
    assert credits[0]["token_advantages"] == [1] * 32 + [-0.5] * 32 + [0.5]
    assert credits[1]["token_advantages"] == [-1] * 32 + [0.5] * 32 + [-0.5]
    with pytest.raises(AssertionError):
        assemble_credits(roots, jobs[:-1], 2)


def test_adjacent_values_are_not_cumulative_returns_or_length_means():
    credit = process_token_credit(
        9,
        1,
        0.25,
        [
            {"position": 2, "value": 0.75, "half_values": [0.5, 1.0]},
            {"position": 5, "value": 0.5, "half_values": [0.25, 0.75]},
        ],
    )
    assert credit["token_advantages"] == [0.5] * 2 + [-0.25] * 3 + [0.5] * 4
    assert sum(b["advantage"] for b in credit["blocks"]) == pytest.approx(0.75)
    # A short response with no interior estimate reduces to outcome-minus-baseline.
    assert process_token_credit(3, 0, 0.75, [])["token_advantages"] == [-0.75] * 3


@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.real_rl
def test_native_packed_loss_keeps_signed_blocks_and_trajectory_denominator(
    temperature, real_rl_stack
):
    import torch
    from pathlib import Path
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl.utils import omega_conf_to_dataclass
    from rllab.real_rl_runtime import compose_config, policy_batch

    repo = Path(__file__).resolve().parents[1]
    cfg = compose_config(
        repo,
        repo / "configs/rl/math_native.yaml",
        {
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 8,
                    "optim": {"lr": 0.0001, "total_training_steps": 1},
                }
            }
        },
    )
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    credit = [[0.5, 0.5, -0.25, -0.25, -0.25, 0.5, 0.5, 0.5, 0.5], [-0.75] * 3]
    outputs = [
        # Frozen raw-token probes may lack rollout log probs. Their placeholders
        # must not reweight or mask the configured plain REINFORCE loss.
        SimpleNamespace(token_ids=[8] * 9, log_probs=[0.0] * 9),
        SimpleNamespace(token_ids=[9] * 3, log_probs=[0.0] * 3),
    ]
    data = policy_batch([[5, 6], [7]], outputs, credit, 0, temperature=temperature)
    from verl.utils import tensordict_utils as tu

    assert tu.get_non_tensor_data(data, "temperature", default=None) == temperature
    assert data["input_ids"].offsets().diff().tolist() == [11, 4]
    logp = torch.full_like(data["advantages"], -2.0, requires_grad=True)
    loss, _ = get_policy_loss_fn(actor.policy_loss.loss_mode)(
        old_log_prob=data["old_log_probs"],
        log_prob=logp,
        advantages=data["advantages"],
        response_mask=data["response_mask"],
        loss_agg_mode=actor.loss_agg_mode,
        config=actor,
    )
    # Total credit sums cancel, but their signed per-token gradient must remain.
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(
        logp.grad, -data["advantages"] * data["response_mask"] / 2
    )
