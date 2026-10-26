"""Batch weighting and scalar reduction for frozen gradient captures."""

from collections import Counter
import math


def aggregate_capture(receipts):
    assert sorted(r["rank"] for r in receipts) == list(range(len(receipts)))
    names = set(receipts[0]["local_dots"])
    assert all(set(r["local_dots"]) == names for r in receipts)
    return dict(
        norm2=math.fsum(r["local_norm2"] for r in receipts),
        dots={
            name: math.fsum(r["local_dots"][name] for r in receipts)
            for name in sorted(names)
        },
        parameters=round(sum(r["weighted_parameter_numel"] for r in receipts)),
    )


def active_question_batch(rows, credits, *, multiple=1):
    """Omit zero-question computation while preserving the original mean loss."""
    assert len(rows) == len(credits) and rows
    assert multiple > 0 and len(rows) % multiple == 0
    active = {
        r["question_id"]
        for r, c in zip(rows, credits, strict=True)
        if any(c["token_advantages"])
    }
    counts = Counter(r["question_id"] for r in rows)
    kept_count = sum(counts[q] for q in active)
    for qid in dict.fromkeys(r["question_id"] for r in rows):
        if kept_count % multiple == 0:
            break
        if qid not in active:
            active.add(qid)
            kept_count += counts[qid]
    kept = [
        (r, c) for r, c in zip(rows, credits, strict=True) if r["question_id"] in active
    ]
    if not kept:
        # Legacy signal captures sometimes explicitly request a zero gradient.
        kept = list(zip(rows, credits, strict=True))
    scale = len(kept) / len(rows)
    return (
        [r for r, _ in kept],
        [[a * scale for a in c["token_advantages"]] for _, c in kept],
        scale,
    )
