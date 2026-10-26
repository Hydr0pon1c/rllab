from __future__ import annotations

from pathlib import Path

from rllab.artifacts import (
    JsonlGzipWriter,
    iter_jsonl_gz,
    load_mdp_from_dataset,
    verify_dataset,
)


def test_deterministic_jsonl_gzip(tmp_path):
    records = ({"id": 1, "text": "甲"}, {"id": 2, "text": "乙"})
    paths = (tmp_path / "left.jsonl.gz", tmp_path / "right.jsonl.gz")
    for path in paths:
        with JsonlGzipWriter(path) as writer:
            for record in records:
                writer.write(record)
        assert writer.count == 2
        assert tuple(iter_jsonl_gz(path)) == records
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_published_smoke_dataset_is_deeply_valid():
    latest = Path(__file__).resolve().parents[1] / "data" / "smoke" / "latest"
    if not latest.is_dir():
        return
    report = verify_dataset(latest)
    assert report["ok"]
    mdp = load_mdp_from_dataset(latest, "smoke-00000")
    assert len(mdp.states) == 13
    assert len(mdp.transitions) == 36
