"""Operational CLI for storage, model snapshots, task probes, and datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from rllab import __version__
from rllab.artifacts import build_dataset, list_datasets, verify_dataset
from rllab.config import load_dataset_config
from rllab.errors import RLLabError
from rllab.mdp.difficulty import synthesize_mdp
from rllab.models import (
    fetch_model_snapshot,
    inspect_model_snapshot,
    register_model_snapshot,
)
from rllab.storage import init_storage, resolve_storage_root, storage_doctor

QWEN3_8B_SOURCE = "Qwen/Qwen3-8B"
QWEN3_8B_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _add_storage_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage-root",
        help="Persistent root (default: RLLAB_STORAGE_ROOT or project default)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rllab-data",
        description="RLLab PuzzleMDP data infrastructure",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    storage = commands.add_parser(
        "storage", help="Manage persistent storage and local links"
    )
    storage_commands = storage.add_subparsers(dest="storage_command", required=True)
    storage_init = storage_commands.add_parser(
        "init", help="Create layout and repo symlinks"
    )
    _add_storage_root(storage_init)
    storage_init.add_argument("--repo-root")
    storage_init.add_argument("--no-links", action="store_true")
    storage_doctor_parser = storage_commands.add_parser(
        "doctor", help="Validate storage layout"
    )
    _add_storage_root(storage_doctor_parser)
    storage_doctor_parser.add_argument("--repo-root")

    model = commands.add_parser(
        "model", help="Fetch and register pinned model snapshots"
    )
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_fetch = model_commands.add_parser(
        "fetch", help="Download a complete HF snapshot"
    )
    _add_storage_root(model_fetch)
    model_fetch.add_argument("--name", default="Qwen3-8B")
    model_fetch.add_argument("--source", default=QWEN3_8B_SOURCE)
    model_fetch.add_argument("--revision", default=QWEN3_8B_REVISION)
    model_fetch.add_argument(
        "--direct",
        action="store_true",
        help="Unset proxy variables only for the download process",
    )
    model_fetch.add_argument("--endpoint", help="Optional Hugging Face mirror endpoint")
    model_inspect = model_commands.add_parser(
        "inspect", help="Validate a local snapshot"
    )
    _add_storage_root(model_inspect)
    model_inspect.add_argument("path")
    model_inspect.add_argument("--name", default="Qwen3-8B")
    model_inspect.add_argument("--source", default=QWEN3_8B_SOURCE)
    model_inspect.add_argument("--revision", default=QWEN3_8B_REVISION)
    model_inspect.add_argument("--register", action="store_true")

    task = commands.add_parser(
        "task", help="Probe task synthesis without tokenizer/model use"
    )
    task_commands = task.add_subparsers(dest="task_command", required=True)
    task_probe = task_commands.add_parser(
        "probe", help="Print realised difficulty metrics"
    )
    task_probe.add_argument("--config", required=True)
    task_probe.add_argument("--count", type=int, default=8)

    dataset = commands.add_parser("dataset", help="Build and verify immutable datasets")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_commands.add_parser(
        "build", help="Build/reuse a dataset version"
    )
    dataset_build.add_argument("--config", required=True)
    _add_storage_root(dataset_build)
    dataset_build.add_argument("--repo-root")
    dataset_verify = dataset_commands.add_parser(
        "verify", help="Verify hashes and row counts"
    )
    dataset_verify.add_argument("path")
    dataset_list = dataset_commands.add_parser(
        "list", help="List published dataset versions"
    )
    _add_storage_root(dataset_list)
    return parser


def _handle(args: argparse.Namespace) -> int:
    if args.command == "storage":
        if args.storage_command == "init":
            _json_print(
                init_storage(
                    args.storage_root,
                    repo_root=args.repo_root,
                    create_links=not args.no_links,
                )
            )
            return 0
        report = storage_doctor(args.storage_root, repo_root=args.repo_root)
        _json_print(report)
        return 0 if report["ok"] else 1

    if args.command == "model":
        if args.model_command == "fetch":
            path, registry = fetch_model_snapshot(
                name=args.name,
                source=args.source,
                revision=args.revision,
                storage_root=args.storage_root,
                direct=args.direct,
                endpoint=args.endpoint,
            )
            _json_print({"path": str(path), "registry": str(registry)})
            return 0
        report = inspect_model_snapshot(
            args.path,
            source=args.source,
            revision=args.revision,
        )
        if args.register:
            registry = register_model_snapshot(
                args.path,
                name=args.name,
                source=args.source,
                revision=args.revision,
                storage_root=args.storage_root,
            )
            report["registry"] = str(registry)
        _json_print(report)
        return 0 if report["complete"] else 1

    if args.command == "task":
        config = load_dataset_config(args.config)
        if args.count < 1:
            raise RLLabError("--count must be positive")
        rows = []
        for index, sequence in enumerate(np_seed_sequences(config.seed, args.count)):
            mdp, report, attempts = synthesize_mdp(
                config.generator,
                config.difficulty,
                seed=sequence,
                mdp_id=f"probe-{index:05d}",
            )
            rows.append(
                {
                    "mdp_id": mdp.mdp_id,
                    "synthesis_seed": sequence,
                    "attempts": attempts,
                    **report.to_dict(),
                }
            )
        numeric_keys = (
            "uniform_regret_per_step",
            "mean_q_gap",
            "delayed_choice_rate",
            "mean_depth_entropy",
            "never_reconverge_rate",
        )
        summary = {
            key: {
                "min": min(float(row[key]) for row in rows),
                "mean": sum(float(row[key]) for row in rows) / len(rows),
                "max": max(float(row[key]) for row in rows),
            }
            for key in numeric_keys
        }
        _json_print(
            {
                "config": str(Path(args.config).resolve()),
                "summary": summary,
                "tasks": rows,
            }
        )
        return 0

    if args.command == "dataset":
        if args.dataset_command == "build":
            config = load_dataset_config(args.config)
            init_storage(args.storage_root, repo_root=args.repo_root, create_links=True)
            path = build_dataset(
                config,
                storage_root=args.storage_root,
                repo_root=args.repo_root,
            )
            _json_print({"path": str(path), "verification": verify_dataset(path)})
            return 0
        if args.dataset_command == "verify":
            _json_print(verify_dataset(args.path))
            return 0
        _json_print(
            {
                "storage_root": str(resolve_storage_root(args.storage_root)),
                "datasets": list_datasets(args.storage_root),
            }
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


def np_seed_sequences(seed: int, count: int) -> tuple[int, ...]:
    # Local import keeps `--help` cheap and makes the seed derivation explicit.
    import numpy as np

    return tuple(
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in np.random.SeedSequence(seed).spawn(count)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _handle(args)
    except RLLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
