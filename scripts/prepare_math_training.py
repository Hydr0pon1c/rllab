"""Freeze paired math plans, train-only startups or fixed-question seed repeats."""

import argparse
import json
from pathlib import Path

from rllab.math_io import atomic_json, fingerprint, now, provenance
from rllab.math_training import build_plan, repeat_plan, startup_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--model", type=Path, help="Local pinned model snapshot")
    parser.add_argument("--dataset", type=Path, help="Normalized questions.jsonl")
    parser.add_argument("--allocation", choices=["dynamic", "uniform24", "uniform32"])
    parser.add_argument("--devices", type=int, choices=[8, 16], default=8)
    parser.add_argument(
        "--reference-plan",
        type=Path,
        help="Require identical questions and schedule across actors",
    )
    derivation = parser.add_mutually_exclusive_group()
    derivation.add_argument(
        "--startup-from",
        type=Path,
        help="Derive train-only startup from an existing formal plan",
    )
    derivation.add_argument(
        "--repeat-from",
        type=Path,
        help="Repeat a formal plan with identical questions and a new training seed",
    )
    parser.add_argument("--seed", type=int, help="Training seed for --repeat-from")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.seed is not None) != bool(args.repeat_from):
        parser.error("--seed and --repeat-from must be supplied together")
    if args.startup_from or args.repeat_from:
        if (
            args.profile
            or args.reference_plan
            or args.model
            or args.dataset
            or args.allocation
        ):
            parser.error(
                "Plan derivation cannot be combined with profile/reference overrides"
            )
        parent = args.repeat_from or args.startup_from
        original = json.loads(parent.read_text())
        plan = (
            repeat_plan(original, args.seed)
            if args.repeat_from
            else startup_plan(original)
        )
        plan["repeat_parent" if args.repeat_from else "startup_parent"] = dict(
            path=str(parent.resolve()), sha256=fingerprint(parent)
        )
    else:
        import yaml
        from transformers import AutoTokenizer

        repo = Path(__file__).resolve().parents[1]
        study = yaml.safe_load((repo / "configs/rl/qwen_math.yaml").read_text())[
            "study"
        ]
        if args.profile:
            study.update(yaml.safe_load(args.profile.read_text())["study"])
        for key in ("model", "dataset"):
            value = getattr(args, key) or study[key]
            study[key] = str(Path(value).expanduser().resolve())
        tokenizer = AutoTokenizer.from_pretrained(study["model"], local_files_only=True)
        reference = (
            json.loads(args.reference_plan.read_text()) if args.reference_plan else None
        )
        plan = build_plan(study, tokenizer, reference=reference)
        if args.reference_plan:
            plan["question_selection_reference"] = dict(
                path=str(args.reference_plan.resolve()),
                sha256=fingerprint(args.reference_plan),
            )
        if args.allocation:
            from rllab.math_mc_cross_risk_training import make_training_plan

            plan = make_training_plan(plan, args.allocation, args.devices)
            plan.update(created_at=now(), **provenance())
    args.output.mkdir(parents=True)
    atomic_json(args.output / "plan.json", plan)
    print(
        json.dumps(
            dict(
                path=str(args.output / "plan.json"),
                counts=plan["counts"],
                study=plan["study"],
            )
        )
    )


if __name__ == "__main__":
    main()
