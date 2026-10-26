"""Persistent-disk layout and safe local symlink management."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

from rllab.errors import ArtifactError

STORAGE_SCHEMA_VERSION = "rllab-storage/v1"
DEFAULT_STORAGE_ROOT = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rllab"
STORAGE_ENV = "RLLAB_STORAGE_ROOT"
STORAGE_DIRS = (
    "datasets",
    "models",
    "geometry",
    "runs",
    "task-logs",
    "cache/huggingface",
    "registry/models",
    "tmp",
    "locks",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_storage_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get(STORAGE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_STORAGE_ROOT


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        existing = link.resolve(strict=False)
        if existing == target:
            return
        raise ArtifactError(
            f"refusing to replace existing symlink {link} -> {existing}"
        )
    if link.exists():
        raise ArtifactError(f"refusing to replace existing path {link}")
    temporary = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    try:
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def init_storage(
    root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    create_links: bool = True,
) -> dict[str, object]:
    resolved_root = resolve_storage_root(root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    for relative in STORAGE_DIRS:
        (resolved_root / relative).mkdir(parents=True, exist_ok=True)

    layout_file = resolved_root / "storage-layout.json"
    if layout_file.exists():
        try:
            existing = json.loads(layout_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"cannot read existing storage layout {layout_file}: {exc}"
            ) from exc
        if existing.get("schema_version") != STORAGE_SCHEMA_VERSION:
            raise ArtifactError(f"unsupported storage layout at {layout_file}")
    else:
        layout = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Large RLLab datasets, model snapshots, caches, and run outputs",
        }
        _atomic_write_text(
            layout_file,
            json.dumps(layout, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    links: dict[str, str] = {}
    if create_links:
        repo = Path(repo_root).resolve() if repo_root is not None else repository_root()
        mapping = {
            "data": resolved_root / "datasets",
            "models": resolved_root / "models",
        }
        for name, target in mapping.items():
            _ensure_link(repo / name, target)
            links[str(repo / name)] = str(target.resolve())
    return {
        "root": str(resolved_root),
        "links": links,
        "schema_version": STORAGE_SCHEMA_VERSION,
    }


@contextmanager
def storage_lock(root: str | Path, name: str) -> Iterator[None]:
    """Advisory per-artifact lock to prevent concurrent partial publishes."""

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        yield
        return
    lock_path = Path(root) / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def storage_doctor(
    root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    resolved_root = resolve_storage_root(root)
    repo = Path(repo_root).resolve() if repo_root is not None else repository_root()
    checks: dict[str, object] = {
        "root": str(resolved_root),
        "root_exists": resolved_root.is_dir(),
        "layout_exists": (resolved_root / "storage-layout.json").is_file(),
        "directories": {
            relative: (resolved_root / relative).is_dir() for relative in STORAGE_DIRS
        },
        "links": {},
    }
    expected = {"data": resolved_root / "datasets", "models": resolved_root / "models"}
    for name, target in expected.items():
        link = repo / name
        checks["links"][name] = {
            "path": str(link),
            "is_symlink": link.is_symlink(),
            "target": str(link.resolve(strict=False)) if link.is_symlink() else None,
            "correct": link.is_symlink()
            and link.resolve(strict=False) == target.resolve(),
        }
    checks["ok"] = bool(
        checks["root_exists"]
        and checks["layout_exists"]
        and all(checks["directories"].values())
        and all(item["correct"] for item in checks["links"].values())
    )
    return checks
