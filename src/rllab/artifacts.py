"""Atomic, checksummed, normalized dataset artifact builds."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from rllab import __version__
from rllab.config import DatasetConfig
from rllab.data.prompts import build_all_prompt_records
from rllab.data.relabel import make_labeling
from rllab.data.tokens import TokenRegistry, build_token_registry, load_tokenizer
from rllab.data.truth import exact_truth_records
from rllab.errors import ArtifactError, ValidationError
from rllab.mdp.difficulty import synthesize_mdp
from rllab.mdp.generate import layer_widths
from rllab.mdp.schema import PuzzleMDP, State, Transition
from rllab.storage import repository_root, resolve_storage_root, storage_lock

DATASET_MANIFEST_SCHEMA_VERSION = "rllab-dataset/v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class JsonlGzipWriter:
    """Streaming deterministic JSONL gzip writer (mtime=0, atomic publish)."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self._temporary: Path | None = None
        self._raw = None
        self._gzip = None
        self._text = None

    def __enter__(self) -> "JsonlGzipWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        self._temporary = Path(temporary_name)
        self._raw = os.fdopen(descriptor, "wb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=self._raw,
            mtime=0,
        )
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="\n")
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self._text is None:
            raise RuntimeError("JSONL writer is not open")
        self._text.write(canonical_json_bytes(dict(record)).decode("utf-8"))
        self.count += 1

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._temporary is not None and self._raw is not None
        try:
            if self._text is not None:
                self._text.flush()
                self._text.detach()
            if self._gzip is not None:
                self._gzip.close()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            if exc_type is None:
                os.replace(self._temporary, self.path)
        finally:
            if not self._raw.closed:
                self._raw.close()
            if self._temporary.exists():
                self._temporary.unlink()


def iter_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactError(
                    f"record at {path}:{line_number} is not a JSON object"
                )
            yield value


def source_tree_fingerprint(repo: str | Path) -> str:
    root = Path(repo).resolve()
    digest = sha256()
    paths = sorted((root / "src" / "rllab").rglob("*.py"))
    if not paths:
        raise ArtifactError(f"cannot fingerprint RLLab sources under {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_build_id(
    config: DatasetConfig,
    registry_fingerprint: str,
    code_fingerprint: str,
) -> str:
    identity = {
        "manifest_schema": DATASET_MANIFEST_SCHEMA_VERSION,
        "code_version": __version__,
        "code_fingerprint": code_fingerprint,
        "config": config.to_dict(),
        "token_registry": registry_fingerprint,
    }
    return sha256(canonical_json_bytes(identity)).hexdigest()[:16]


def _split_assignments(config: DatasetConfig) -> tuple[str, ...]:
    names = ("train", "validation", "test")
    fractions = np.asarray(
        [config.splits.train, config.splits.validation, config.splits.test],
        dtype=np.float64,
    )
    raw_counts = fractions * config.num_instances
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = config.num_instances - int(counts.sum())
    priorities = sorted(
        range(len(names)),
        key=lambda index: (-(raw_counts[index] - counts[index]), index),
    )
    for index in priorities[:remainder]:
        counts[index] += 1
    labels = [
        name
        for name, count in zip(names, counts, strict=True)
        for _ in range(int(count))
    ]
    rng = np.random.default_rng(config.seed ^ 0x5A17)
    rng.shuffle(labels)
    return tuple(labels)


def _resolve_tokenizer_path(config: DatasetConfig, repo: Path) -> Path:
    configured = Path(config.tokenizer.path).expanduser()
    resolved = (
        configured.resolve()
        if configured.is_absolute()
        else (repo / configured).resolve()
    )
    if not resolved.is_dir():
        raise ArtifactError(
            f"tokenizer path does not exist: {resolved}; initialise storage and fetch/register the model"
        )
    return resolved


def _table_metadata(path: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "records": records,
    }


def _update_latest_link(dataset_family: Path, build_id: str) -> None:
    link = dataset_family / "latest"
    temporary = dataset_family / f".latest.tmp-{os.getpid()}"
    try:
        temporary.symlink_to(build_id, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def build_dataset(
    config: DatasetConfig,
    *,
    storage_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Path:
    """Build or reuse one immutable content-addressed dataset version."""

    config.validate()
    storage = resolve_storage_root(storage_root)
    repo = Path(repo_root).resolve() if repo_root is not None else repository_root()
    tokenizer_path = _resolve_tokenizer_path(config, repo)
    tokenizer = load_tokenizer(tokenizer_path)
    required_states = sum(layer_widths(config.generator))
    if config.tokenizer.state_pool_size < required_states:
        raise ValidationError(
            f"state_pool_size={config.tokenizer.state_pool_size} is smaller than "
            f"the generated state count {required_states}"
        )
    registry = build_token_registry(
        tokenizer,
        tokenizer_path=tokenizer_path,
        source=config.tokenizer.source,
        revision=config.tokenizer.revision,
        action_labels=config.tokenizer.action_labels,
        state_pool_size=config.tokenizer.state_pool_size,
        pool_seed=config.seed,
        enable_thinking=config.tokenizer.enable_thinking,
    )
    code_fingerprint = source_tree_fingerprint(repo)
    build_id = _dataset_build_id(config, registry.fingerprint, code_fingerprint)
    dataset_family = storage / "datasets" / config.name
    target = dataset_family / build_id

    with storage_lock(storage, f"dataset-{config.name}"):
        if target.exists():
            verify_dataset(target)
            _update_latest_link(dataset_family, build_id)
            return target

        dataset_family.mkdir(parents=True, exist_ok=True)
        temporary = dataset_family / f".{build_id}.tmp-{os.getpid()}"
        if temporary.exists():
            raise ArtifactError(f"temporary build path already exists: {temporary}")
        temporary.mkdir()
        try:
            _atomic_write_bytes(
                temporary / "config.json", canonical_json_bytes(config.to_dict())
            )
            _atomic_write_bytes(
                temporary / "token_registry.json",
                canonical_json_bytes(registry.to_dict()),
            )

            table_names = (
                "instances",
                "states",
                "transitions",
                "labelings",
                "prompts",
                "truth",
            )
            writers = {
                name: JsonlGzipWriter(temporary / f"{name}.jsonl.gz")
                for name in table_names
            }
            entered: list[JsonlGzipWriter] = []
            try:
                for writer in writers.values():
                    writer.__enter__()
                    entered.append(writer)

                split_assignments = _split_assignments(config)
                instance_sequences = np.random.SeedSequence(config.seed).spawn(
                    config.num_instances
                )
                for instance_index, (sequence, split) in enumerate(
                    zip(instance_sequences, split_assignments, strict=True)
                ):
                    instance_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
                    mdp_id = f"{config.name}-{instance_index:05d}"
                    mdp, difficulty, attempts = synthesize_mdp(
                        config.generator,
                        config.difficulty,
                        seed=instance_seed,
                        mdp_id=mdp_id,
                    )
                    writers["instances"].write(
                        {
                            "schema_version": "dataset-instance/v1",
                            "mdp_id": mdp.mdp_id,
                            "split": split,
                            "seed": mdp.seed,
                            "synthesis_seed": instance_seed,
                            "synthesis_attempts": attempts,
                            "start_state": mdp.start_state,
                            "horizon": mdp.horizon,
                            "num_actions": mdp.num_actions,
                            "metadata": dict(mdp.metadata),
                        }
                    )
                    for state in mdp.ordered_states():
                        writers["states"].write(
                            {
                                "schema_version": "puzzle-state/v1",
                                "mdp_id": mdp.mdp_id,
                                "split": split,
                                **asdict(state),
                            }
                        )
                    for transition in sorted(
                        mdp.transitions, key=lambda item: (item.source, item.action)
                    ):
                        writers["transitions"].write(
                            {
                                "schema_version": "puzzle-transition/v1",
                                "mdp_id": mdp.mdp_id,
                                "split": split,
                                **asdict(transition),
                            }
                        )
                    for truth in exact_truth_records(mdp):
                        writers["truth"].write({"split": split, **truth})

                    labeling_sequences = sequence.spawn(len(config.labeling_modes))
                    for mode, labeling_sequence in zip(
                        config.labeling_modes, labeling_sequences, strict=True
                    ):
                        labeling_seed = int(
                            labeling_sequence.generate_state(1, dtype=np.uint64)[0]
                        )
                        labeling = make_labeling(
                            mdp,
                            registry,
                            mode=mode,
                            seed=labeling_seed,
                            labeling_id=f"{mdp.mdp_id}-{mode}",
                        )
                        writers["labelings"].write(
                            {"split": split, **labeling.to_dict()}
                        )
                        prompts = build_all_prompt_records(
                            mdp,
                            labeling,
                            registry,
                            tokenizer,
                            enable_thinking=config.tokenizer.enable_thinking,
                        )
                        for prompt in prompts:
                            writers["prompts"].write(
                                {"split": split, **prompt.to_dict()}
                            )
            except BaseException:
                for writer in reversed(entered):
                    writer.__exit__(*__import__("sys").exc_info())
                raise
            else:
                for writer in reversed(entered):
                    writer.__exit__(None, None, None)

            tables = [
                _table_metadata(temporary / f"{name}.jsonl.gz", writers[name].count)
                for name in table_names
            ]
            static_files = [
                _table_metadata(temporary / "config.json", 1),
                _table_metadata(temporary / "token_registry.json", 1),
            ]
            manifest = {
                "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
                "dataset_name": config.name,
                "build_id": build_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code_version": __version__,
                "code_fingerprint": code_fingerprint,
                "token_registry_fingerprint": registry.fingerprint,
                "source_model": {
                    "repo_id": config.tokenizer.source,
                    "revision": config.tokenizer.revision,
                    "tokenizer_fingerprint": registry.tokenizer_fingerprint,
                },
                "counts": {
                    "instances": config.num_instances,
                    "labeling_modes": list(config.labeling_modes),
                    "states_per_instance": required_states,
                },
                "files": static_files + tables,
            }
            _atomic_write_bytes(
                temporary / "manifest.json", canonical_json_bytes(manifest)
            )
            os.replace(temporary, target)
            _update_latest_link(dataset_family, build_id)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    verify_dataset(target)
    return target


def verify_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"cannot read dataset manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported dataset manifest schema at {manifest_path}")
    errors: list[str] = []
    for entry in manifest.get("files", []):
        try:
            file_path = (root / entry["path"]).resolve()
        except (KeyError, TypeError, OSError) as exc:
            errors.append(f"invalid manifest file entry: {exc}")
            continue
        if root != file_path.parent:
            errors.append(f"manifest file escapes dataset root: {entry.get('path')!r}")
            continue
        if not file_path.is_file():
            errors.append(f"missing {entry['path']}")
            continue
        if file_path.stat().st_size != entry["bytes"]:
            errors.append(f"size mismatch for {entry['path']}")
        if sha256_file(file_path) != entry["sha256"]:
            errors.append(f"sha256 mismatch for {entry['path']}")
        if entry["path"].endswith(".jsonl.gz"):
            try:
                count = sum(1 for _ in iter_jsonl_gz(file_path))
            except (OSError, ArtifactError) as exc:
                errors.append(f"cannot read {entry['path']}: {exc}")
            else:
                if count != entry["records"]:
                    errors.append(
                        f"record count mismatch for {entry['path']}: {count} != {entry['records']}"
                    )
    if not errors:
        errors.extend(_relational_integrity_errors(root, manifest))
    if errors:
        raise ArtifactError("dataset verification failed: " + "; ".join(errors))
    return {
        "ok": True,
        "path": str(root),
        "build_id": manifest["build_id"],
        "dataset_name": manifest["dataset_name"],
        "files": len(manifest.get("files", [])),
    }


def _relational_integrity_errors(root: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Check cross-table keys, coverage, token positions, and MDP invariants."""

    errors: list[str] = []
    try:
        registry_raw = json.loads(
            (root / "token_registry.json").read_text(encoding="utf-8")
        )
        registry = TokenRegistry.from_dict(registry_raw)
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
        return [f"invalid token_registry.json: {exc}"]
    if registry.fingerprint != manifest.get("token_registry_fingerprint"):
        errors.append("manifest token registry fingerprint mismatch")

    try:
        instance_rows = list(iter_jsonl_gz(root / "instances.jsonl.gz"))
        state_rows = list(iter_jsonl_gz(root / "states.jsonl.gz"))
        transition_rows = list(iter_jsonl_gz(root / "transitions.jsonl.gz"))
        labeling_rows = list(iter_jsonl_gz(root / "labelings.jsonl.gz"))
    except (OSError, ArtifactError) as exc:
        return [f"cannot load normalized tables: {exc}"]

    instances: dict[str, Mapping[str, Any]] = {}
    for row in instance_rows:
        mdp_id = str(row.get("mdp_id"))
        if mdp_id in instances:
            errors.append(f"duplicate instance {mdp_id}")
        instances[mdp_id] = row
    expected_instances = manifest.get("counts", {}).get("instances")
    if expected_instances is not None and len(instances) != expected_instances:
        errors.append(f"instance cardinality {len(instances)} != {expected_instances}")

    states_by_mdp: dict[str, list[Mapping[str, Any]]] = {
        mdp_id: [] for mdp_id in instances
    }
    transitions_by_mdp: dict[str, list[Mapping[str, Any]]] = {
        mdp_id: [] for mdp_id in instances
    }
    for table_name, rows, destination in (
        ("states", state_rows, states_by_mdp),
        ("transitions", transition_rows, transitions_by_mdp),
    ):
        for row in rows:
            mdp_id = str(row.get("mdp_id"))
            if mdp_id not in instances:
                errors.append(f"{table_name} row references unknown MDP {mdp_id}")
                continue
            if row.get("split") != instances[mdp_id].get("split"):
                errors.append(f"{table_name} split mismatch for {mdp_id}")
            destination[mdp_id].append(row)

    mdps: dict[str, PuzzleMDP] = {}
    for mdp_id, instance in instances.items():
        try:
            raw = {
                "schema_version": "puzzle-mdp/v1",
                "mdp_id": mdp_id,
                "seed": instance["seed"],
                "start_state": instance["start_state"],
                "horizon": instance["horizon"],
                "num_actions": instance["num_actions"],
                "states": [
                    {
                        key: row[key]
                        for key in ("state_id", "depth", "terminal", "terminal_cost")
                    }
                    for row in states_by_mdp[mdp_id]
                ],
                "transitions": [
                    {key: row[key] for key in ("source", "action", "target", "cost")}
                    for row in transitions_by_mdp[mdp_id]
                ],
                "metadata": instance.get("metadata", {}),
            }
            mdps[mdp_id] = PuzzleMDP.from_dict(raw)
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            errors.append(f"invalid canonical MDP {mdp_id}: {exc}")

    registry_action_ids = tuple(token.token_id for token in registry.action_tokens)
    registry_state_ids = {token.token_id for token in registry.state_tokens}
    labelings: dict[str, dict[str, Any]] = {}
    for row in labeling_rows:
        labeling_id = str(row.get("labeling_id"))
        mdp_id = str(row.get("mdp_id"))
        if labeling_id in labelings:
            errors.append(f"duplicate labeling {labeling_id}")
            continue
        if mdp_id not in mdps:
            errors.append(
                f"labeling {labeling_id} references invalid/unknown MDP {mdp_id}"
            )
            continue
        if row.get("split") != instances[mdp_id].get("split"):
            errors.append(f"labeling split mismatch for {labeling_id}")
        if row.get("registry_fingerprint") != registry.fingerprint:
            errors.append(f"registry fingerprint mismatch for {labeling_id}")
        raw_mapping = row.get("state_tokens", {})
        canonical_states = {state.state_id for state in mdps[mdp_id].states}
        if set(raw_mapping) != canonical_states:
            errors.append(f"state coverage mismatch for labeling {labeling_id}")
        try:
            mapping_ids = {
                state_id: int(token["token_id"])
                for state_id, token in raw_mapping.items()
            }
            action_ids = tuple(int(token["token_id"]) for token in row["action_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed token mapping for {labeling_id}: {exc}")
            continue
        if len(set(mapping_ids.values())) != len(mapping_ids):
            errors.append(f"state token IDs are not one-to-one for {labeling_id}")
        if set(mapping_ids.values()) - registry_state_ids:
            errors.append(
                f"labeling {labeling_id} uses IDs outside the registry state pool"
            )
        if action_ids != registry_action_ids:
            errors.append(f"action token mismatch for labeling {labeling_id}")
        labelings[labeling_id] = {
            "mdp_id": mdp_id,
            "mapping_ids": mapping_ids,
            "action_ids": action_ids,
            "seen_states": set(),
        }

    seen_examples: set[str] = set()
    try:
        for row in iter_jsonl_gz(root / "prompts.jsonl.gz"):
            example_id = str(row.get("example_id"))
            labeling_id = str(row.get("labeling_id"))
            state_id = str(row.get("state_id"))
            if example_id in seen_examples:
                errors.append(f"duplicate prompt example {example_id}")
            seen_examples.add(example_id)
            if labeling_id not in labelings:
                errors.append(f"prompt references unknown labeling {labeling_id}")
                continue
            info = labelings[labeling_id]
            mdp = mdps[info["mdp_id"]]
            state_by_id = {state.state_id: state for state in mdp.states}
            if state_id not in state_by_id or state_by_id[state_id].terminal:
                errors.append(
                    f"prompt references invalid current state {labeling_id}:{state_id}"
                )
                continue
            if state_id in info["seen_states"]:
                errors.append(f"duplicate state prompt {labeling_id}:{state_id}")
            info["seen_states"].add(state_id)
            if row.get("mdp_id") != mdp.mdp_id:
                errors.append(f"prompt MDP mismatch for {example_id}")
            if row.get("split") != instances[mdp.mdp_id].get("split"):
                errors.append(f"prompt split mismatch for {example_id}")
            input_ids = [int(value) for value in row.get("input_ids", [])]
            if tuple(row.get("allowed_action_token_ids", [])) != info["action_ids"]:
                errors.append(f"allowed action IDs mismatch for {example_id}")
            expected_sha = sha256(
                json.dumps(input_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest()
            if expected_sha != row.get("prompt_sha256"):
                errors.append(f"prompt hash mismatch for {example_id}")
            role_counts = {"current_state": 0, "action": 0, "destination_state": 0}
            for slot in row.get("slots", []):
                role = slot.get("role")
                if role in role_counts:
                    role_counts[role] += 1
                try:
                    position = int(slot["token_position"])
                    canonical_id = str(slot["canonical_id"])
                    expected_id = (
                        info["action_ids"][int(canonical_id)]
                        if role == "action"
                        else info["mapping_ids"][canonical_id]
                    )
                    actual_id = input_ids[position]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    errors.append(f"malformed slot in {example_id}: {exc}")
                    continue
                if actual_id != expected_id:
                    errors.append(f"slot token mismatch in {example_id} at {position}")
            expected_roles = {
                "current_state": 1,
                "action": mdp.num_actions,
                "destination_state": mdp.num_actions,
            }
            if role_counts != expected_roles:
                errors.append(
                    f"placeholder role counts mismatch for {example_id}: {role_counts}"
                )
    except (OSError, ArtifactError) as exc:
        errors.append(f"cannot validate prompts: {exc}")

    for labeling_id, info in labelings.items():
        mdp = mdps[info["mdp_id"]]
        expected_states = {state.state_id for state in mdp.states if not state.terminal}
        if info["seen_states"] != expected_states:
            errors.append(f"prompt state coverage mismatch for {labeling_id}")

    truth_seen: dict[str, set[str]] = {mdp_id: set() for mdp_id in mdps}
    try:
        for row in iter_jsonl_gz(root / "truth.jsonl.gz"):
            mdp_id = str(row.get("mdp_id"))
            state_id = str(row.get("state_id"))
            if mdp_id not in mdps:
                errors.append(f"truth row references unknown MDP {mdp_id}")
                continue
            if state_id in truth_seen[mdp_id]:
                errors.append(f"duplicate truth row {mdp_id}:{state_id}")
            truth_seen[mdp_id].add(state_id)
            state = next(
                (item for item in mdps[mdp_id].states if item.state_id == state_id),
                None,
            )
            if state is None:
                errors.append(f"truth row references unknown state {mdp_id}:{state_id}")
                continue
            expected_actions = 0 if state.terminal else mdps[mdp_id].num_actions
            if len(row.get("actions", [])) != expected_actions:
                errors.append(f"truth action count mismatch for {mdp_id}:{state_id}")
            if row.get("split") != instances[mdp_id].get("split"):
                errors.append(f"truth split mismatch for {mdp_id}:{state_id}")
    except (OSError, ArtifactError) as exc:
        errors.append(f"cannot validate truth: {exc}")
    for mdp_id, seen in truth_seen.items():
        expected = {state.state_id for state in mdps[mdp_id].states}
        if seen != expected:
            errors.append(f"truth state coverage mismatch for {mdp_id}")
    return errors


def load_mdp_from_dataset(path: str | Path, mdp_id: str) -> PuzzleMDP:
    root = Path(path).resolve()
    instance = next(
        (
            record
            for record in iter_jsonl_gz(root / "instances.jsonl.gz")
            if record["mdp_id"] == mdp_id
        ),
        None,
    )
    if instance is None:
        raise ArtifactError(f"MDP {mdp_id!r} is not present in {root}")
    states = tuple(
        State(
            state_id=record["state_id"],
            depth=int(record["depth"]),
            terminal=bool(record["terminal"]),
            terminal_cost=int(record["terminal_cost"]),
        )
        for record in iter_jsonl_gz(root / "states.jsonl.gz")
        if record["mdp_id"] == mdp_id
    )
    transitions = tuple(
        Transition(
            source=record["source"],
            action=int(record["action"]),
            target=record["target"],
            cost=int(record["cost"]),
        )
        for record in iter_jsonl_gz(root / "transitions.jsonl.gz")
        if record["mdp_id"] == mdp_id
    )
    raw = {
        "schema_version": "puzzle-mdp/v1",
        "mdp_id": mdp_id,
        "seed": instance["seed"],
        "start_state": instance["start_state"],
        "horizon": instance["horizon"],
        "num_actions": instance["num_actions"],
        "states": [asdict(state) for state in states],
        "transitions": [asdict(transition) for transition in transitions],
        "metadata": instance.get("metadata", {}),
    }
    return PuzzleMDP.from_dict(raw)


def list_datasets(storage_root: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    datasets_root = resolve_storage_root(storage_root) / "datasets"
    if not datasets_root.is_dir():
        return ()
    results: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for manifest_path in sorted(datasets_root.glob("*/*/manifest.json")):
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest in seen:
            continue
        seen.add(resolved_manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results.append(
            {
                "dataset_name": manifest.get("dataset_name"),
                "build_id": manifest.get("build_id"),
                "created_at": manifest.get("created_at"),
                "path": str(resolved_manifest.parent),
                "instances": manifest.get("counts", {}).get("instances"),
            }
        )
    return tuple(results)
