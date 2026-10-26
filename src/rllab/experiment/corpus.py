"""Read immutable PuzzleMDP tables into experiment-oriented bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rllab.artifacts import iter_jsonl_gz, load_mdp_from_dataset, verify_dataset
from rllab.errors import ArtifactError, ValidationError
from rllab.mdp.schema import PuzzleMDP


@dataclass(frozen=True)
class PromptExample:
    example_id: str
    mdp_id: str
    labeling_id: str
    state_id: str
    depth: int
    input_ids: tuple[int, ...]
    prompt_sha256: str
    action_token_ids: tuple[int, ...]
    current_state_token_position: int
    state_slots: tuple[tuple[int, str, str], ...]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "PromptExample":
        input_ids = tuple(int(value) for value in row["input_ids"])
        state_slots: list[tuple[int, str, str]] = []
        current_positions: list[int] = []
        for slot in row["slots"]:
            role = str(slot["role"])
            if role not in {"current_state", "destination_state"}:
                continue
            position = int(slot["token_position"])
            canonical_id = str(slot["canonical_id"])
            if not 0 <= position < len(input_ids):
                raise ArtifactError(
                    f"slot outside prompt {row.get('example_id')}: {position}"
                )
            state_slots.append((position, role, canonical_id))
            if role == "current_state":
                current_positions.append(position)
        if len(current_positions) != 1:
            raise ArtifactError(
                f"prompt {row.get('example_id')} needs exactly one current-state slot"
            )
        return cls(
            example_id=str(row["example_id"]),
            mdp_id=str(row["mdp_id"]),
            labeling_id=str(row["labeling_id"]),
            state_id=str(row["state_id"]),
            depth=int(row["depth"]),
            input_ids=input_ids,
            prompt_sha256=str(row["prompt_sha256"]),
            action_token_ids=tuple(
                int(value) for value in row["allowed_action_token_ids"]
            ),
            current_state_token_position=current_positions[0],
            state_slots=tuple(state_slots),
        )


@dataclass(frozen=True)
class LabelingInfo:
    labeling_id: str
    mdp_id: str
    mode: str
    state_token_ids: Mapping[str, int]
    action_token_ids: tuple[int, ...]
    source: str
    assignment_score: float | None = None
    assignment_metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InstanceBundle:
    mdp: PuzzleMDP
    split: str
    labeling: LabelingInfo
    prompts: Mapping[str, PromptExample]
    truth: Mapping[str, Mapping[str, Any]]

    @property
    def nonterminal_state_ids(self) -> tuple[str, ...]:
        return tuple(
            state.state_id for state in self.mdp.ordered_states() if not state.terminal
        )

    def ordered_prompts(self) -> tuple[PromptExample, ...]:
        return tuple(self.prompts[state_id] for state_id in self.nonterminal_state_ids)


@dataclass(frozen=True)
class CorpusIdentity:
    dataset_path: str
    dataset_name: str
    dataset_build_id: str
    dataset_manifest_sha256: str
    geometry_path: str | None
    geometry_build_id: str | None
    token_registry_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "dataset_build_id": self.dataset_build_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "geometry_path": self.geometry_path,
            "geometry_build_id": self.geometry_build_id,
            "token_registry_fingerprint": self.token_registry_fingerprint,
        }


@dataclass(frozen=True)
class ExperimentCorpus:
    identity: CorpusIdentity
    bundles: tuple[InstanceBundle, ...]
    action_token_ids: tuple[int, ...]
    prompt_length: int
    labeling_mode: str
    split: str

    @property
    def example_count(self) -> int:
        return sum(len(bundle.prompts) for bundle in self.bundles)

    def all_prompts(self) -> tuple[PromptExample, ...]:
        return tuple(
            prompt for bundle in self.bundles for prompt in bundle.ordered_prompts()
        )

    def bundle_by_mdp(self) -> dict[str, InstanceBundle]:
        return {bundle.mdp.mdp_id: bundle for bundle in self.bundles}


def _sha256(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_optional_geometry(
    geometry_path: str | Path | None,
) -> tuple[
    Path | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]
]:
    if geometry_path is None:
        return None, None, [], []
    root = Path(geometry_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"cannot load geometry manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != "rllab-geometry/v1":
        raise ArtifactError(f"unsupported geometry artifact at {root}")
    for entry in manifest.get("files", []):
        path = root / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise ArtifactError(f"geometry checksum failed for {path}")
    return (
        root,
        manifest,
        list(iter_jsonl_gz(root / "labelings.jsonl.gz")),
        list(iter_jsonl_gz(root / "prompts.jsonl.gz")),
    )


def _labeling_info(row: Mapping[str, Any], *, source: str) -> LabelingInfo:
    try:
        state_token_ids = {
            str(state_id): int(token["token_id"])
            for state_id, token in row["state_tokens"].items()
        }
        action_ids = tuple(int(token["token_id"]) for token in row["action_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(
            f"malformed labeling {row.get('labeling_id')}: {exc}"
        ) from exc
    return LabelingInfo(
        labeling_id=str(row["labeling_id"]),
        mdp_id=str(row["mdp_id"]),
        mode=str(row["mode"]),
        state_token_ids=state_token_ids,
        action_token_ids=action_ids,
        source=source,
        assignment_score=(
            float(row["assignment_score"])
            if row.get("assignment_score") is not None
            else None
        ),
        assignment_metadata=dict(row.get("assignment_metadata", {})),
    )


def load_experiment_corpus(
    dataset_path: str | Path,
    *,
    split: str,
    labeling_mode: str,
    geometry_path: str | Path | None = None,
    max_instances: int | None = None,
) -> ExperimentCorpus:
    """Load one paired labeling condition without materialising tensor weights."""

    root = Path(dataset_path).expanduser().resolve()
    verify_dataset(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    geometry_root, geometry_manifest, geometry_labelings, geometry_prompts = (
        _load_optional_geometry(geometry_path)
    )
    if geometry_manifest is not None:
        if geometry_manifest.get("base_dataset_build_id") != manifest["build_id"]:
            raise ArtifactError("geometry artifact was built from a different dataset")
        if (
            geometry_manifest.get("token_registry_fingerprint")
            != manifest["token_registry_fingerprint"]
        ):
            raise ArtifactError("geometry token registry differs from the base dataset")

    instance_rows = list(iter_jsonl_gz(root / "instances.jsonl.gz"))
    selected_ids = sorted(
        str(row["mdp_id"]) for row in instance_rows if str(row["split"]) == split
    )
    if max_instances is not None:
        selected_ids = selected_ids[:max_instances]
    if not selected_ids:
        raise ValidationError(f"dataset has no instances for split={split!r}")
    instance_by_id = {str(row["mdp_id"]): row for row in instance_rows}

    labeling_rows = (
        list(iter_jsonl_gz(root / "labelings.jsonl.gz")) + geometry_labelings
    )
    labelings: dict[str, LabelingInfo] = {}
    chosen_by_mdp: dict[str, LabelingInfo] = {}
    for row in labeling_rows:
        source = "geometry" if row in geometry_labelings else "dataset"
        info = _labeling_info(row, source=source)
        if info.labeling_id in labelings:
            raise ArtifactError(
                f"duplicate labeling ID across artifacts: {info.labeling_id}"
            )
        labelings[info.labeling_id] = info
        if info.mdp_id not in selected_ids or info.mode != labeling_mode:
            continue
        if info.mdp_id in chosen_by_mdp:
            raise ArtifactError(
                f"multiple {labeling_mode!r} labelings for MDP {info.mdp_id}"
            )
        chosen_by_mdp[info.mdp_id] = info
    missing = set(selected_ids) - set(chosen_by_mdp)
    if missing:
        hint = (
            " (provide data.geometry_path for measured modes)"
            if geometry_path is None
            else ""
        )
        raise ArtifactError(
            f"labeling mode {labeling_mode!r} missing for MDPs {sorted(missing)}{hint}"
        )

    prompt_rows = list(iter_jsonl_gz(root / "prompts.jsonl.gz")) + geometry_prompts
    prompts_by_labeling: dict[str, dict[str, PromptExample]] = {
        info.labeling_id: {} for info in chosen_by_mdp.values()
    }
    for row in prompt_rows:
        labeling_id = str(row.get("labeling_id"))
        if labeling_id not in prompts_by_labeling:
            continue
        prompt = PromptExample.from_row(row)
        if prompt.state_id in prompts_by_labeling[labeling_id]:
            raise ArtifactError(f"duplicate prompt {labeling_id}:{prompt.state_id}")
        prompts_by_labeling[labeling_id][prompt.state_id] = prompt

    truth_by_mdp: dict[str, dict[str, Mapping[str, Any]]] = {
        mdp_id: {} for mdp_id in selected_ids
    }
    for row in iter_jsonl_gz(root / "truth.jsonl.gz"):
        mdp_id = str(row["mdp_id"])
        if mdp_id in truth_by_mdp:
            truth_by_mdp[mdp_id][str(row["state_id"])] = row

    bundles: list[InstanceBundle] = []
    for mdp_id in selected_ids:
        mdp = load_mdp_from_dataset(root, mdp_id)
        labeling = chosen_by_mdp[mdp_id]
        prompts = prompts_by_labeling[labeling.labeling_id]
        expected = {state.state_id for state in mdp.states if not state.terminal}
        if set(prompts) != expected:
            raise ArtifactError(
                f"prompt coverage mismatch for {labeling.labeling_id}: "
                f"missing={sorted(expected - set(prompts))}"
            )
        if set(labeling.state_token_ids) != {state.state_id for state in mdp.states}:
            raise ArtifactError(
                f"state-token coverage mismatch for {labeling.labeling_id}"
            )
        bundles.append(
            InstanceBundle(
                mdp=mdp,
                split=str(instance_by_id[mdp_id]["split"]),
                labeling=labeling,
                prompts=prompts,
                truth=truth_by_mdp[mdp_id],
            )
        )

    action_id_sets = {
        prompt.action_token_ids
        for bundle in bundles
        for prompt in bundle.prompts.values()
    }
    lengths = {
        len(prompt.input_ids)
        for bundle in bundles
        for prompt in bundle.prompts.values()
    }
    if len(action_id_sets) != 1 or len(lengths) != 1:
        raise ArtifactError(
            f"experiment requires invariant action IDs/prompt lengths, got {action_id_sets}/{lengths}"
        )
    action_ids = next(iter(action_id_sets))
    if len(set(action_ids)) != len(action_ids):
        raise ArtifactError("action token IDs are not unique")

    identity = CorpusIdentity(
        dataset_path=str(root),
        dataset_name=str(manifest["dataset_name"]),
        dataset_build_id=str(manifest["build_id"]),
        dataset_manifest_sha256=_sha256(manifest_path),
        geometry_path=str(geometry_root) if geometry_root is not None else None,
        geometry_build_id=(
            str(geometry_manifest["build_id"])
            if geometry_manifest is not None
            else None
        ),
        token_registry_fingerprint=str(manifest["token_registry_fingerprint"]),
    )
    return ExperimentCorpus(
        identity=identity,
        bundles=tuple(bundles),
        action_token_ids=action_ids,
        prompt_length=next(iter(lengths)),
        labeling_mode=labeling_mode,
        split=split,
    )


def prompts_for_rank(
    prompts: Iterable[PromptExample], *, rank: int, world_size: int
) -> tuple[PromptExample, ...]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world_size")
    return tuple(
        prompt for index, prompt in enumerate(prompts) if index % world_size == rank
    )
