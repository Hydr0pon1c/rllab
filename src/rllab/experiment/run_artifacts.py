"""Transactional run metadata, metrics, and resumable full-precision checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from rllab.artifacts import JsonlGzipWriter, canonical_json_bytes
from rllab.errors import ArtifactError
from rllab.experiment.config import ExperimentConfig
from rllab.experiment.corpus import ExperimentCorpus
from rllab.experiment.modeling import unwrap_policy
from rllab.experiment.runtime import DistributedContext, barrier
from rllab.storage import repository_root, storage_lock


RUN_SCHEMA_VERSION = "rllab-run/v1"
_ACTIVE_RUN: RunDirectory | None = None


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(dict(value)))


def _git(command: list[str], root: Path) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    root = repository_root()
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain=v1", "--untracked-files=all"], root)
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise ArtifactError(
            "refusing to launch a scientific run from a dirty worktree; commit changes "
            "or pass --allow-dirty only for local debugging"
        )
    return {
        "commit": commit,
        "dirty": dirty,
        "status": status.splitlines(),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
    }


def config_fingerprint(config: ExperimentConfig, corpus: ExperimentCorpus) -> str:
    identity = {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": config.to_dict(),
        "corpus": corpus.identity.to_dict(),
    }
    # Resume is operational rather than scientific identity.
    identity["config"]["output"]["resume"] = False
    identity["config"]["output"]["run_id"] = None
    return sha256(canonical_json_bytes(identity)).hexdigest()


@dataclass(frozen=True)
class RunDirectory:
    path: Path
    run_id: str
    config_fingerprint: str

    @property
    def calibration_path(self) -> Path:
        return self.path / "calibration.jsonl.gz"

    @property
    def checkpoints_path(self) -> Path:
        return self.path / "checkpoints"

    def append_metric(self, row: Mapping[str, Any]) -> None:
        path = self.path / "metrics.jsonl"
        payload = canonical_json_bytes(dict(row))
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def write_records(self, relative: str, rows: Iterable[Mapping[str, Any]]) -> Path:
        path = self.path / relative
        with JsonlGzipWriter(path) as writer:
            for row in rows:
                writer.write(row)
        return path

    def update_status(self, status: str, **extra: Any) -> None:
        atomic_json(
            self.path / "status.json",
            {
                "schema_version": "rllab-run-status/v1",
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
        )


def _set_active_run(run: RunDirectory | None) -> None:
    global _ACTIVE_RUN
    _ACTIVE_RUN = run


def clear_active_run(run: RunDirectory) -> None:
    """Stop CLI-level failure handling from changing a completed run."""

    if _ACTIVE_RUN == run:
        _set_active_run(None)


def mark_active_run_terminal(
    status: str,
    error: BaseException,
    *,
    context: DistributedContext | None,
) -> None:
    """Best-effort terminal marker for failures outside the training loop."""

    run = _ACTIVE_RUN
    if run is None or (context is not None and not context.is_main):
        return
    preserved: dict[str, Any] = {}
    status_path = run.path / "status.json"
    try:
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed":
                return
            for key in ("completed_steps", "target_steps"):
                if key in previous:
                    preserved[key] = previous[key]
        run.update_status(
            status,
            **preserved,
            error={
                "type": type(error).__name__,
                "message": str(error)[:4000],
                "rank": context.rank if context is not None else 0,
            },
        )
    except Exception:  # pragma: no cover - never mask the original failure
        return


def prepare_run_directory(
    config: ExperimentConfig,
    corpus: ExperimentCorpus,
    *,
    context: DistributedContext,
    model_fingerprint: str,
    runtime_reports: list[Mapping[str, Any]],
    allow_dirty: bool,
) -> RunDirectory:
    fingerprint = config_fingerprint(config, corpus)
    run_id = config.output.run_id or (
        f"{config.data.labeling_mode}-{config.training.credit_mode}-"
        f"s{config.runtime.seed}-{fingerprint[:10]}"
    )
    path = (
        Path(config.output.root).expanduser().resolve()
        / config.output.run_name
        / run_id
    )
    if context.is_main:
        provenance = git_provenance(allow_dirty=allow_dirty)
        lock_root = Path(config.output.root).expanduser().resolve()
        lock_root.mkdir(parents=True, exist_ok=True)
        with storage_lock(lock_root, f"run-{config.output.run_name}-{run_id}"):
            if path.exists():
                if not config.output.resume:
                    raise ArtifactError(
                        f"run directory already exists: {path}; use a new run ID or --resume"
                    )
                existing = json.loads((path / "run.json").read_text(encoding="utf-8"))
                if existing.get("config_fingerprint") != fingerprint:
                    raise ArtifactError("resume config differs from the existing run")
            else:
                path.mkdir(parents=True)
                atomic_json(path / "config.resolved.json", config.to_dict())
                atomic_json(
                    path / "run.json",
                    {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "run_id": run_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "config_fingerprint": fingerprint,
                        "model_fingerprint": model_fingerprint,
                        "corpus": corpus.identity.to_dict(),
                        "selection": {
                            "split": corpus.split,
                            "labeling_mode": corpus.labeling_mode,
                            "instances": len(corpus.bundles),
                            "prompts": corpus.example_count,
                            "prompt_length": corpus.prompt_length,
                            "action_token_ids": list(corpus.action_token_ids),
                        },
                        "git": provenance,
                        "runtime_by_rank": runtime_reports,
                    },
                )
            (path / "checkpoints").mkdir(exist_ok=True)
    barrier(context)
    if not path.is_dir():
        raise ArtifactError(f"rank {context.rank} cannot see run directory {path}")
    run = RunDirectory(path=path, run_id=run_id, config_fingerprint=fingerprint)
    _set_active_run(run)
    return run


def latest_checkpoint(run: RunDirectory) -> Path | None:
    link = run.checkpoints_path / "latest"
    if not link.is_symlink():
        return None
    resolved = link.resolve()
    if not resolved.is_dir() or resolved.parent != run.checkpoints_path:
        raise ArtifactError(f"invalid latest checkpoint link: {link}")
    return resolved


def save_checkpoint(
    run: RunDirectory,
    policy,
    optimizer,
    *,
    completed_steps: int,
    generator,
    context: DistributedContext,
    keep: int,
) -> Path:
    import torch

    name = f"step-{completed_steps:08d}"
    temporary = run.checkpoints_path / f".{name}.tmp"
    target = run.checkpoints_path / name
    if context.is_main:
        if temporary.exists() or target.exists():
            raise ArtifactError(f"checkpoint path already exists for {name}")
        temporary.mkdir()
    barrier(context)

    rank_state = {
        "schema_version": "rllab-rng-state/v1",
        "rank": context.rank,
        "world_size": context.world_size,
        "completed_steps": completed_steps,
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "rollout_generator": generator.get_state(),
    }
    torch.save(rank_state, temporary / f"rank-{context.rank:05d}.pt")
    barrier(context)
    if context.is_main:
        module = unwrap_policy(policy)
        model_path = temporary / "model"
        module.model.save_pretrained(
            model_path,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        atomic_json(
            temporary / "trainer_state.json",
            {
                "schema_version": "rllab-checkpoint/v1",
                "completed_steps": completed_steps,
                "world_size": context.world_size,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        os.replace(temporary, target)
        link = run.checkpoints_path / "latest"
        temporary_link = run.checkpoints_path / ".latest.tmp"
        if temporary_link.is_symlink():
            temporary_link.unlink()
        temporary_link.symlink_to(name, target_is_directory=True)
        os.replace(temporary_link, link)

        checkpoints = sorted(
            (
                path
                for path in run.checkpoints_path.glob("step-*")
                if path.is_dir() and path != target
            ),
            key=lambda path: path.name,
        )
        for old in checkpoints[: max(0, len(checkpoints) - keep + 1)]:
            shutil.rmtree(old)
    barrier(context)
    return target


def restore_training_state(
    checkpoint: Path,
    optimizer,
    *,
    generator,
    context: DistributedContext,
) -> int:
    import torch

    trainer_state = json.loads(
        (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
    )
    if int(trainer_state["world_size"]) != context.world_size:
        raise ArtifactError(
            "resume requires the same world size to preserve trajectory RNG streams"
        )
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cuda", weights_only=False)
    )
    state = torch.load(
        checkpoint / f"rank-{context.rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"].cpu())
    generator.set_state(state["rollout_generator"].cpu())
    return int(trainer_state["completed_steps"])
