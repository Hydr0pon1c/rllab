"""One Qwen policy path shared by calibration, rollout, loss, and evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from rllab.errors import ArtifactError, RLLabError
from rllab.experiment.config import ModelConfig
from rllab.experiment.runtime import DistributedContext


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - PPU-only boundary
        raise RLLabError("PyTorch is required for the PPU experiment runtime") from exc
    return torch


def model_snapshot_fingerprint(path: str | Path, revision: str) -> str:
    """Hash model identity without re-reading every multi-gigabyte shard."""

    root = Path(path).expanduser().resolve()
    names = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    digest = sha256()
    digest.update(revision.encode("utf-8"))
    for name in names:
        file_path = root / name
        if not file_path.is_file():
            raise ArtifactError(f"model snapshot is missing {file_path}")
        payload = file_path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    for shard in sorted(set(index["weight_map"].values())):
        shard_path = root / shard
        if not shard_path.is_file():
            raise ArtifactError(f"model snapshot is missing shard {shard_path}")
        digest.update(shard.encode("utf-8"))
        digest.update(shard_path.stat().st_size.to_bytes(8, "big"))
    return digest.hexdigest()


class MaskedActionPolicy:
    """Thin wrapper around a CausalLM that exposes only allowed action logits.

    The class dynamically subclasses ``torch.nn.Module`` through composition to
    keep importing CPU data utilities possible without importing torch. Use
    :func:`load_policy` to construct the actual module instance.
    """


def _policy_module_class():
    torch = _torch()

    class _MaskedActionPolicy(torch.nn.Module):
        def __init__(self, model, action_token_ids: Iterable[int], compute_dtype):
            super().__init__()
            self.model = model
            action_ids = torch.tensor(tuple(action_token_ids), dtype=torch.long)
            self.register_buffer("action_token_ids", action_ids, persistent=False)
            self.compute_dtype = compute_dtype

        def forward(self, input_ids, attention_mask=None):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.compute_dtype == torch.bfloat16,
            ):
                output = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            # The vocabulary projection is row-separable. Selecting the four
            # action rows before the matmul is exactly equivalent to projecting
            # all 151,936 tokens and gathering afterwards, while avoiding a
            # large unused output. Keep this small scientific boundary in FP32
            # even when transformer matmuls use BF16 autocast.
            final_hidden = output.last_hidden_state[:, -1, :].float()
            action_weight = self.model.lm_head.weight.index_select(
                0, self.action_token_ids
            )
            return torch.nn.functional.linear(final_hidden, action_weight.float())

        def log_probs(self, input_ids, fixed_bias, attention_mask=None):
            logits = self.forward(input_ids, attention_mask=attention_mask)
            if fixed_bias.shape != logits.shape:
                raise ValueError(
                    f"fixed bias shape {tuple(fixed_bias.shape)} != logits {tuple(logits.shape)}"
                )
            return torch.log_softmax(logits + fixed_bias.float(), dim=-1)

        def probabilities(self, input_ids, fixed_bias, attention_mask=None):
            return self.log_probs(input_ids, fixed_bias, attention_mask).exp()

    return _MaskedActionPolicy


def load_policy(config: ModelConfig, action_token_ids: tuple[int, ...]):
    torch = _torch()
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - PPU-only boundary
        raise RLLabError("transformers is missing from the PPU image") from exc

    model_path = Path(config.path).expanduser().resolve()
    hf_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if hf_config.model_type != "qwen3":
        raise RLLabError(f"expected a qwen3 snapshot, got {hf_config.model_type!r}")
    if bool(hf_config.tie_word_embeddings):
        raise RLLabError("RLLab requires untied input/output embeddings")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=config.attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    module_class = _policy_module_class()
    compute_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[config.compute_dtype]
    policy = module_class(model, action_token_ids, compute_dtype)
    return policy.cuda()


def wrap_distributed(policy, context: DistributedContext):
    if not context.distributed:
        return policy
    torch = _torch()
    return torch.nn.parallel.DistributedDataParallel(
        policy,
        device_ids=[context.device_index],
        output_device=context.device_index,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


def unwrap_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def no_sync_context(policy, *, enabled: bool):
    if enabled and hasattr(policy, "no_sync"):
        return policy.no_sync()
    return nullcontext()


def policy_parameter_report(policy) -> dict[str, Any]:
    torch = _torch()
    module = unwrap_policy(policy)
    parameters = tuple(module.parameters())
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    dtype_counts: dict[str, int] = {}
    for parameter in trainable:
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
    if not trainable or any(
        parameter.dtype != torch.float32 for parameter in trainable
    ):
        raise RLLabError("all trainable parameters must be FP32 master parameters")
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "trainable_dtype_counts": dtype_counts,
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in parameters
        ),
    }
