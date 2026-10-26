"""Fast PPU image, kernel, and NCCL diagnostics."""

from __future__ import annotations

import time
from typing import Any

from rllab.experiment.config import ExperimentConfig
from rllab.experiment.corpus import load_experiment_corpus
from rllab.experiment.modeling import (
    load_policy,
    policy_parameter_report,
    unwrap_policy,
    wrap_distributed,
)
from rllab.experiment.runtime import (
    DistributedContext,
    assert_device_capacity,
    distributed_collective_probe,
    gather_objects,
    runtime_report,
)


def run_doctor(
    config: ExperimentConfig,
    *,
    context: DistributedContext,
    model_probe: bool,
) -> dict[str, Any]:
    import torch

    report = runtime_report(context)
    report["capacity"] = assert_device_capacity(
        config.runtime.min_free_device_gib, context
    )
    report["collective"] = distributed_collective_probe(context)

    size = 4096
    left = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        product = left @ right
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10):
        product = left @ right
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report["bf16_matmul"] = {
        "shape": [size, size],
        "iterations": 10,
        "seconds": elapsed,
        "checksum": float(product.float().mean().item()),
    }
    del left, right, product
    torch.cuda.empty_cache()

    if model_probe:
        corpus = load_experiment_corpus(
            config.data.path,
            split=config.data.split,
            labeling_mode=config.data.labeling_mode,
            geometry_path=config.data.geometry_path,
            max_instances=1,
        )
        prompt = corpus.bundles[0].ordered_prompts()[0]
        load_started = time.perf_counter()
        policy = wrap_distributed(
            load_policy(config.model, corpus.action_token_ids), context
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        input_ids = torch.tensor(
            [prompt.input_ids] * 4, dtype=torch.long, device="cuda"
        )
        policy.train()
        forward_started = time.perf_counter()
        logits = policy(input_ids)
        fixed_bias = logits.detach().mean(dim=1, keepdim=True) - logits.detach()
        log_probs = torch.log_softmax(logits + fixed_bias, dim=-1)
        loss = -log_probs[:, context.rank % len(corpus.action_token_ids)].mean()
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started
        backward_started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.perf_counter() - backward_started
        module = unwrap_policy(policy)
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1.0e-7)
        torch.cuda.synchronize()
        report["qwen_train_probe"] = {
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "loss": float(loss.item()),
            "logits": logits.detach().cpu().tolist(),
            "parameters": policy_parameter_report(policy),
            "memory": {
                "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            },
        }
    reports = gather_objects(report, context)
    return {"ok": True, "world_size": context.world_size, "ranks": reports}
