"""Command line entry point for PPU diagnostics and scientific runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from typing import Sequence

from rllab.errors import RLLabError
from rllab.experiment.config import load_experiment_config
from rllab.experiment.run_artifacts import mark_active_run_terminal
from rllab.experiment.runtime import initialize_distributed, shutdown_distributed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rllab-exp",
        description="PuzzleMDP full-parameter training and exact evaluation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="Validate PPU kernels and NCCL")
    doctor.add_argument("--config", required=True)
    doctor.add_argument(
        "--model-probe",
        action="store_true",
        help="Also run a full Qwen3-8B FP32-master forward/backward probe",
    )

    train = commands.add_parser("train", help="Run one ORM/PRM × labeling condition")
    train.add_argument("--config", required=True)
    train.add_argument("--run-id")
    train.add_argument(
        "--seed",
        type=int,
        help="Override runtime.seed for a checked-in multi-seed study",
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Local debugging only; platform runs should require a clean commit",
    )

    geometry = commands.add_parser(
        "geometry", help="Build measured aligned/adversarial relabel overlays"
    )
    geometry.add_argument("--config", required=True)
    geometry.add_argument("--name", required=True)
    geometry.add_argument(
        "--output-root",
        default="outputs/geometry",
    )
    geometry.add_argument(
        "--assignment-scope",
        choices=("full", "depth"),
        default="full",
        help="Constrain state-token matching globally or within canonical depth",
    )
    geometry.add_argument("--allow-dirty", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = None
    try:
        config = load_experiment_config(args.config)
        if args.command == "train" and args.run_id:
            output = replace(
                config.output,
                run_id=args.run_id,
                resume=bool(args.resume or config.output.resume),
            )
            config = replace(config, output=output)
        if args.command == "train" and (args.resume or args.seed is not None):
            output = replace(
                config.output,
                resume=bool(args.resume or config.output.resume),
            )
            runtime = (
                replace(config.runtime, seed=args.seed)
                if args.seed is not None
                else config.runtime
            )
            config = replace(config, output=output, runtime=runtime)
            config.validate()
        context = initialize_distributed()
        if args.command == "doctor":
            from rllab.experiment.doctor import run_doctor

            result = run_doctor(
                config,
                context=context,
                model_probe=args.model_probe,
            )
        elif args.command == "geometry":
            from rllab.experiment.geometry import build_geometry

            result = build_geometry(
                config,
                context=context,
                name=args.name,
                output_root=args.output_root,
                allow_dirty=args.allow_dirty,
                assignment_scope=args.assignment_scope,
            )
        else:
            from rllab.experiment.train import run_training

            result = run_training(
                config,
                context=context,
                allow_dirty=args.allow_dirty,
            )
        if context.is_main:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except RLLabError as exc:
        mark_active_run_terminal("failed", exc, context=context)
        rank = context.rank if context is not None else 0
        print(f"rank {rank} error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt as exc:
        mark_active_run_terminal("interrupted", exc, context=context)
        return 130
    except Exception as exc:
        mark_active_run_terminal("failed", exc, context=context)
        raise
    finally:
        if context is not None:
            shutdown_distributed(context)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
