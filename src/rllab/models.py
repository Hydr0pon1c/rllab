"""Pinned Hugging Face snapshot download and local model registration."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterator

from rllab.errors import ArtifactError
from rllab.storage import resolve_storage_root

MODEL_REGISTRY_SCHEMA_VERSION = "model-snapshot/v1"
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _temporary_network_environment(
    *, direct: bool, endpoint: str | None
) -> Iterator[None]:
    keys = (*PROXY_KEYS, "HF_ENDPOINT")
    old = {key: os.environ.get(key) for key in keys}
    try:
        if direct:
            for key in PROXY_KEYS:
                os.environ.pop(key, None)
        if endpoint:
            os.environ["HF_ENDPOINT"] = endpoint
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def inspect_model_snapshot(
    path: str | Path,
    *,
    source: str,
    revision: str,
) -> dict[str, object]:
    root = Path(path).resolve()
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ArtifactError(f"model snapshot is missing required files: {missing}")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot parse model config under {root}: {exc}") from exc

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
    else:
        shard_names = (
            ["model.safetensors"] if (root / "model.safetensors").is_file() else []
        )
    missing_shards = [name for name in shard_names if not (root / name).is_file()]
    if missing_shards:
        raise ArtifactError(
            f"model snapshot has missing weight shards: {missing_shards}"
        )
    shards = [
        {"name": name, "size": (root / name).stat().st_size} for name in shard_names
    ]
    metadata_revisions: set[str] = set()
    for name in (*required, *shard_names):
        metadata_path = (
            root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        )
        if not metadata_path.is_file():
            continue
        first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if first_line:
            metadata_revisions.add(first_line)
    if metadata_revisions and metadata_revisions != {revision}:
        raise ArtifactError(
            f"snapshot metadata revisions {sorted(metadata_revisions)} do not match {revision}"
        )
    return {
        "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
        "source": source,
        "revision": revision,
        "metadata_revisions": sorted(metadata_revisions),
        "revision_verified": metadata_revisions == {revision},
        "path": str(root),
        "inspected_at": datetime.now(timezone.utc).isoformat(),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "config_sha256": _sha256_file(root / "config.json"),
        "tokenizer_sha256": _sha256_file(root / "tokenizer.json"),
        "weight_shards": shards,
        "weight_bytes": sum(int(item["size"]) for item in shards),
        "complete": bool(shards) and not missing_shards,
    }


def register_model_snapshot(
    path: str | Path,
    *,
    name: str,
    source: str,
    revision: str,
    storage_root: str | Path | None = None,
) -> Path:
    manifest = inspect_model_snapshot(path, source=source, revision=revision)
    registry_path = (
        resolve_storage_root(storage_root) / "registry" / "models" / f"{name}.json"
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_name(f".{registry_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, registry_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return registry_path


def fetch_model_snapshot(
    *,
    name: str,
    source: str,
    revision: str,
    storage_root: str | Path | None = None,
    direct: bool = False,
    endpoint: str | None = None,
) -> tuple[Path, Path]:
    root = resolve_storage_root(storage_root)
    target = root / "models" / name
    cache = root / "cache" / "huggingface"
    target.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with _temporary_network_environment(direct=direct, endpoint=endpoint):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ArtifactError(
                "huggingface-hub is required for model downloads; install rllab-puzzle[tokenizer]"
            ) from exc
        snapshot_download(
            repo_id=source,
            revision=revision,
            local_dir=target,
            cache_dir=cache,
        )
    registry = register_model_snapshot(
        target,
        name=name,
        source=source,
        revision=revision,
        storage_root=root,
    )
    return target, registry
