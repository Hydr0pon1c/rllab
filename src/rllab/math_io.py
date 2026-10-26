"""JSON records and source provenance shared by math training and analysis.

Writers keep the existing math artifact format. Unlike PuzzleMDP's canonical,
fsynced run store, these helpers assume an existing parent and a single writer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, overwrite=False):
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with Path(path).open("w" if overwrite else "x") as output:
        output.write(payload)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ).splitlines(),
    }


def binding(path):
    return dict(path=str(path), sha256=fingerprint(path))
