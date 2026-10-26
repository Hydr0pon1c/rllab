"""Vendor-neutral CUDA-compatible distributed runtime for PPU images."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import random
import socket
from typing import Any

import numpy as np

from rllab.errors import RLLabError


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device_index: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - PPU-only boundary
        raise RLLabError(
            "the experiment runtime needs the platform PPU PyTorch image; "
            "do not install a stock PyPI torch wheel"
        ) from exc
    return torch


def initialize_distributed() -> DistributedContext:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RLLabError("CUDA-compatible PPU runtime is unavailable")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise RLLabError(
            f"invalid distributed environment rank={rank}, world_size={world_size}"
        )
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RLLabError(
            f"LOCAL_RANK={local_rank} is outside {torch.cuda.device_count()} visible devices"
        )
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device_index=local_rank,
    )


def shutdown_distributed(context: DistributedContext) -> None:
    torch = _torch()
    if context.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    torch = _torch()
    if context.distributed:
        torch.distributed.barrier()


def gather_objects(value: Any, context: DistributedContext) -> list[Any]:
    torch = _torch()
    if not context.distributed:
        return [value]
    gathered: list[Any] = [None for _ in range(context.world_size)]
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def broadcast_object(
    value: Any, context: DistributedContext, *, source: int = 0
) -> Any:
    torch = _torch()
    if not context.distributed:
        return value
    values = [value if context.rank == source else None]
    torch.distributed.broadcast_object_list(values, src=source)
    return values[0]


def seed_everything(
    seed: int, context: DistributedContext, *, deterministic: bool
) -> None:
    torch = _torch()
    rank_seed = seed + 1_000_003 * context.rank
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def assert_device_capacity(
    min_free_gib: float, context: DistributedContext
) -> dict[str, float]:
    torch = _torch()
    free, total = torch.cuda.mem_get_info(context.device_index)
    report = {"free_gib": free / 2**30, "total_gib": total / 2**30}
    if report["free_gib"] < min_free_gib:
        raise RLLabError(
            f"device {context.device_index} has only {report['free_gib']:.2f} GiB free; "
            f"the FP32-master Qwen3-8B path requires at least {min_free_gib:.2f} GiB"
        )
    return report


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_report(context: DistributedContext) -> dict[str, Any]:
    torch = _torch()
    properties = torch.cuda.get_device_properties(context.device_index)
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "torch": package_version("torch"),
        "torch_cuda": torch.version.cuda,
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "safetensors": package_version("safetensors"),
        "flash_attn": package_version("flash-attn"),
        "triton": package_version("triton"),
        "device": properties.name,
        "device_total_bytes": properties.total_memory,
        "device_capability": list(
            torch.cuda.get_device_capability(context.device_index)
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "nccl_available": bool(torch.distributed.is_nccl_available()),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PPU_SDK",
                "CUDA_HOME",
                "NCCL_CUMEM_ENABLE",
                "NCCL_CUMEM_HOST_ENABLE",
                "NCCL_IB_DISABLE",
                "PYTORCH_ALLOC_CONF",
            )
        },
    }


def distributed_collective_probe(context: DistributedContext) -> dict[str, Any]:
    torch = _torch()
    value = torch.tensor(float(context.rank + 1), device="cuda", dtype=torch.float32)
    if context.distributed:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = context.world_size * (context.world_size + 1) / 2
    actual = float(value.item())
    if actual != expected:
        raise RLLabError(f"NCCL all-reduce returned {actual}, expected {expected}")
    return {"all_reduce_sum": actual, "expected": expected, "ok": True}
