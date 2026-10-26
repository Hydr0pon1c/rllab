"""Recover the study's fixed MATH questions from the two pinned PRM800K files."""

import argparse
import json
from pathlib import Path

from rllab.math_io import fingerprint, write_json


def prepare(source_dir, output, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    selected = {q["question_id"]: q for q in manifest["questions"]}
    assert len(selected) == len(manifest["questions"])
    questions = {}
    for source in manifest["sources"]:
        path = source_dir / f"{source['name']}.jsonl"
        if fingerprint(path) != source["sha256"]:
            raise ValueError(f"Source differs from the pinned revision: {path}")
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                qid = "math:" + row["unique_id"]
                if qid not in selected:
                    continue
                level = row["level"]
                question = dict(
                    question_id=qid,
                    source="math",
                    original_split=row["unique_id"].split("/")[0],
                    stratum="math_1_2"
                    if level <= 2
                    else ("math_3" if level == 3 else "math_4_5"),
                    level=level,
                    subject=row["subject"],
                    problem=row["problem"],
                    answer=row["answer"],
                    group_id=selected[qid]["group_id"],
                    split=selected[qid]["split"],
                )
                if qid in questions and questions[qid] != question:
                    raise ValueError(f"Conflicting source records: {qid}")
                questions[qid] = question
    if questions.keys() != selected.keys():
        raise ValueError("Pinned sources do not contain every selected question")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "questions.jsonl"
    with destination.open("x") as handle:
        for qid in sorted(questions):
            handle.write(json.dumps(questions[qid], ensure_ascii=False) + "\n")
    write_json(
        output / "manifest.json",
        dict(
            selection_sha256=fingerprint(manifest_path),
            source_revision=manifest["source_revision"],
            sources=manifest["sources"],
            questions_sha256=fingerprint(destination),
            counts={
                split: sum(q["split"] == split for q in questions.values())
                for split in ("train", "test")
            },
            scope="Selected questions only; historical group identities and train/test assignments are preserved.",
        ),
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "configs/rl/math_questions.json"
    print(prepare(args.source_dir, args.output, manifest))


if __name__ == "__main__":
    main()
