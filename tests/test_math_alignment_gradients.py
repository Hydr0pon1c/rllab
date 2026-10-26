"""Frozen-write and distributed geometry checks using the vendor Torch on CPU."""

import pytest

pytestmark = pytest.mark.real_rl


def test_intercepted_sgd_preserves_parameters_and_projects_actual_directions(
    real_rl_stack,
):
    import torch
    from rllab.math_alignment_gradients import FrozenGradientBank

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, -2.0]]))
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1000)
    bank = FrozenGradientBank(model, optimizer, 1)

    def capture(gradient, **kwargs):
        bank.arm(**kwargs)
        (model.weight * torch.tensor([gradient])).sum().backward()
        optimizer.step()
        return bank.finish()

    p = capture([3.0, 4.0], save="prm")
    o = capture([0.0, 5.0], save="orm", references=["prm"])
    assert p["local_norm2"] == o["local_norm2"] == 25
    assert o["local_dots"]["prm"] == 20
    bank.make_directions(dict(prm=5.0, orm=5.0), 0.01)
    result = capture([1.0, 2.0], references=["prm", "orm", "bf16_prm", "bf16_orm"])
    assert result["local_dots"]["prm"] == pytest.approx(2.2)
    assert result["local_dots"]["orm"] == pytest.approx(2.0)
    hypothetical = (original - 0.01 * torch.tensor([[0.6, 0.8]])).bfloat16().float()
    expected = (
        (original.bfloat16().float() - hypothetical) / 0.01 * torch.tensor([[1.0, 2.0]])
    ).sum()
    assert result["local_dots"]["bf16_prm"] == pytest.approx(float(expected))
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    assert model.weight.grad is None and not optimizer.state
    assert bank.intercepted_steps == 3
    bank.close()


def test_signal_sums_and_cross_replicates_do_not_use_noisy_self_norms(real_rl_stack):
    import torch
    from rllab.math_alignment_gradients import FrozenGradientBank

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    bank = FrozenGradientBank(model, optimizer, 1)
    diagonal = 0.0
    for a, b in (([2.0, 1.0], [0.0, -1.0]), ([1.0, 2.0], [-1.0, 0.0])):
        bank.arm(save="question_a", accumulate=["sum_a"])
        (model.weight * torch.tensor([a])).sum().backward()
        optimizer.step()
        bank.finish()
        bank.arm(accumulate=["sum_b"], references=["question_a"])
        (model.weight * torch.tensor([b])).sum().backward()
        optimizer.step()
        diagonal += bank.finish()["local_dots"]["question_a"]
        bank.drop(["question_a"])
    assert diagonal == -2  # Negative cross-signal estimates must survive.
    assert bank.dot("sum_a", "sum_b") == -6
    assert bank.dot("sum_a", "sum_a") == 18
    bank.close()


def test_single_source_direction_preserves_scale_and_freezes_updates(real_rl_stack):
    import torch
    from rllab.math_alignment_gradients import FrozenGradientBank

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, -2.0]]))
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    bank = FrozenGradientBank(model, optimizer, 1)
    bank.arm(save="source")
    (model.weight * torch.tensor([[3.0, 4.0]])).sum().backward()
    optimizer.step()
    bank.finish()
    result = bank.make_directions({"source": 5.0}, 0.01)
    assert result["source"]["local_unit_norm2"] == pytest.approx(1)
    bank.arm(accumulate=["primary"], references=["source", "bf16_source"])
    (model.weight * torch.tensor([[1.0, 2.0]])).sum().backward()
    optimizer.step()
    receipt = bank.finish()
    assert receipt["local_dots"]["source"] == pytest.approx(2.2)
    assert bank.dot("primary", "source") == pytest.approx(2.2)
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    bank.close()


def test_combination_keeps_reference_bank_and_simulates_bf16_without_writes(
    real_rl_stack,
):
    import torch
    from rllab.math_alignment_gradients import FrozenGradientBank

    model = torch.nn.Linear(2, 1, bias=False)
    original = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1000)
    bank = FrozenGradientBank(model, optimizer, 1)
    for name, vector in (
        ("fit", [1.0, 0.0]),
        ("baseline", [3.0, 4.0]),
        ("corrected", [2.0, 2.0]),
    ):
        bank.arm(save=name)
        (model.weight * torch.tensor([vector])).sum().backward()
        optimizer.step()
        bank.finish()
    result = bank.combine("corrected", {"corrected": 1.0, "fit": 1.0})
    assert result["local_norm2"] == 13
    assert bank.dot("corrected", "fit") == bank.dot("baseline", "fit") == 3
    bank.make_directions({"baseline": 5.0, "corrected": 13**0.5}, 0.01)
    assert len(bank.buffers) == 5
    assert bank.dot("fit", "fit") == 1
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    assert not optimizer.state and model.weight.grad is None
    bank.close()


def _distributed_layout(rank, rendezvous):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import (
        DTensor,
        Partial,
        Replicate,
        Shard,
        distribute_tensor,
    )
    from rllab.math_alignment_gradients import host_dot, local_tensor_and_weight

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        for placement in (Shard(0), Replicate()):
            tensor = distribute_tensor(x, mesh, [placement])
            local, weight = local_tensor_and_weight(tensor, 2)
            scalar = torch.tensor(weight * host_dot(local, local), dtype=torch.float64)
            dist.all_reduce(scalar)
            assert scalar.item() == 55
        partial = DTensor.from_local(x, mesh, [Partial()], run_check=False)
        with pytest.raises(AssertionError, match="reduced first"):
            local_tensor_and_weight(partial, 2)
    finally:
        dist.destroy_process_group()


def test_rank_reduction_handles_uneven_shards_and_replicas(real_rl_stack, tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_layout, args=(f"file://{tmp_path}/gloo",), nprocs=2, join=True
    )
