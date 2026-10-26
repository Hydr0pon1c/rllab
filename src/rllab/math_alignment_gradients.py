"""Frozen native-gradient interception and rank-local scalar measurements."""

import math

import torch
from torch.distributed.tensor import DTensor


def local_tensor_and_weight(tensor, world_size):
    """Weight local dots so summing worker receipts counts replicas only once."""
    if not isinstance(tensor, DTensor):
        return tensor, 1.0 / world_size
    assert tensor.device_mesh.size() == world_size
    replicas = 1
    for axis, placement in enumerate(tensor.placements):
        if placement.is_replicate():
            replicas *= tensor.device_mesh.size(axis)
        else:
            assert placement.is_shard(), "Partial gradients must be reduced first"
    return tensor.to_local(), 1.0 / replicas


def host_dot(left, right):
    """FP32 products with FP64 reduction, bounded temporary host memory."""
    assert left.device.type == right.device.type == "cpu"
    assert left.shape == right.shape and left.dtype == right.dtype == torch.float32
    a, b = left.reshape(-1), right.reshape(-1)
    return math.fsum(
        float((a[i : i + 2**22] * b[i : i + 2**22]).sum(dtype=torch.float64))
        for i in range(0, a.numel(), 2**22)
    )


class FrozenGradientBank:
    """A native SGD pre-hook records gradients, then prevents every parameter write.

    References stay as local CPU shards. No full-model gather, gradient file or
    native optimizer replacement is needed. The caller sums scalar rank receipts.
    """

    def __init__(self, module, optimizer, world_size):
        assert type(optimizer) is torch.optim.SGD
        assert not optimizer.state
        assert all(
            g["momentum"] == g["weight_decay"] == 0 and not g["maximize"]
            for g in optimizer.param_groups
        )
        self.parameters = list(module.parameters())
        assert all(p.requires_grad for p in self.parameters)
        assert {id(p) for p in self.parameters} == {
            id(p) for g in optimizer.param_groups for p in g["params"]
        }
        self.world_size = world_size
        layout = [local_tensor_and_weight(p, world_size) for p in self.parameters]
        self.shapes = [t.shape for t, _ in layout]
        self.weights = [w for _, w in layout]
        self.weighted_numel = sum(t.numel() * w for t, w in layout)
        self.buffers = {}
        self.pending = self.result = None
        self.intercepted_steps = 0
        self.hook = optimizer.register_step_pre_hook(self._intercept)

    def arm(self, *, save=None, accumulate=(), references=()):
        assert self.pending is None and self.result is None
        assert save is None or save not in self.buffers
        assert all(name in self.buffers for name in references)
        assert len(set(accumulate)) == len(accumulate)
        assert save not in accumulate
        self.pending = dict(save=save, accumulate=accumulate, references=references)

    def finish(self):
        assert self.pending is None and self.result is not None
        result, self.result = self.result, None
        return result

    def _intercept(self, optimizer, _args, _kwargs):
        try:
            assert self.pending is not None, "Unexpected optimizer step"
            self.result = self._capture(**self.pending)
            self.pending = None
            self.intercepted_steps += 1
        finally:
            # Setting lr=0 alone is unnecessary: empty SGD gradients guarantee
            # no optimizer parameter write, including on a failed capture.
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = None

    def _capture(self, save, accumulate, references):
        if save is not None:
            self.buffers[save] = []
        for name in accumulate:
            self.buffers.setdefault(name, [])
        assert len(self.buffers) <= 5
        norm_terms, missing = [], 0
        dot_terms = {name: [] for name in references}
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                value = torch.zeros(self.shapes[i], dtype=torch.float32)
                missing += 1
            else:
                local, weight = local_tensor_and_weight(parameter.grad, self.world_size)
                assert local.shape == self.shapes[i] and weight == self.weights[i]
                value = local.detach().to(device="cpu", dtype=torch.float32, copy=True)
            norm_terms.append(self.weights[i] * host_dot(value, value))
            for name in references:
                dot_terms[name].append(
                    self.weights[i] * host_dot(value, self.buffers[name][i])
                )
            if save is not None:
                self.buffers[save].append(value)
            for name in accumulate:
                target = self.buffers[name]
                if len(target) == i:
                    target.append(value.clone())
                else:
                    target[i].add_(value)
        result = dict(
            local_norm2=math.fsum(norm_terms),
            local_dots={k: math.fsum(v) for k, v in dot_terms.items()},
            weighted_parameter_numel=self.weighted_numel,
            missing_gradient_parameters=missing,
        )
        assert math.isfinite(result["local_norm2"])
        assert all(math.isfinite(v) for v in result["local_dots"].values())
        return result

    def dot(self, first, second):
        return math.fsum(
            w * host_dot(a, b)
            for w, a, b in zip(
                self.weights, self.buffers[first], self.buffers[second], strict=True
            )
        )

    def combine(self, destination, coefficients):
        """Replace one captured host bank by a linear combination; never write weights."""
        assert self.pending is None and self.result is None
        assert destination in self.buffers and coefficients
        assert set(coefficients) <= set(self.buffers)
        assert all(math.isfinite(v) for v in coefficients.values())
        for i in range(len(self.parameters)):
            value = torch.zeros_like(self.buffers[destination][i])
            for name, coefficient in coefficients.items():
                value.add_(self.buffers[name][i], alpha=coefficient)
            self.buffers[destination][i] = value
        return dict(local_norm2=self.dot(destination, destination))

    def make_directions(self, norms, delta):
        """Keep unit gradients and simulated BF16 descent directions per unit delta."""
        assert set(norms) <= set(self.buffers) and 1 <= len(norms) <= 2
        assert len(self.buffers) + len(norms) <= 5
        assert all(f"bf16_{name}" not in self.buffers for name in norms)
        assert all(not name.startswith("bf16_") for name in norms)
        assert delta > 0
        for name, norm in norms.items():
            assert math.isfinite(norm) and norm > 0
            effective = []
            for parameter, gradient in zip(
                self.parameters, self.buffers[name], strict=True
            ):
                local, _ = local_tensor_and_weight(parameter, self.world_size)
                theta = local.detach().to(device="cpu", dtype=torch.float32)
                updated = torch.add(theta, gradient, alpha=-delta / norm)
                effective.append(
                    (theta.bfloat16().float() - updated.bfloat16().float()) / delta
                )
                gradient.div_(norm)
            self.buffers[f"bf16_{name}"] = effective
        return {
            name: dict(
                local_unit_norm2=self.dot(name, name),
                local_effective_norm2=self.dot(f"bf16_{name}", f"bf16_{name}"),
                local_nominal_effective_dot=self.dot(name, f"bf16_{name}"),
            )
            for name in norms
        }

    def drop(self, names):
        assert self.pending is None and self.result is None
        for name in names:
            del self.buffers[name]

    def close(self):
        self.hook.remove()
        self.buffers.clear()
