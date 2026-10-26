"""Cross-question risk and greedy MC16/32 allocation, independent of the trainer."""

import hashlib
import json
import math

DELTA = 0.01
WALSH = ((1, 1, -1, -1), (1, -1, 1, -1), (1, -1, -1, 1))


def token_credit(positions, values):
    assert len(positions) == len(values) and positions[0] == 0
    assert positions == sorted(set(positions))
    return [
        float(values[j + 1] - values[j])
        for j in range(len(values) - 1)
        for _ in range(positions[j + 1] - positions[j])
    ]


def subset_credits(saved, pools, indices):
    """Only internal values change; root outcome and root LOO stay fixed."""
    assert indices and len(set(indices)) == len(indices)
    assert all(0 <= i < 32 for i in indices)
    result = []
    for c in saved:
        states = c["states"]
        values = [states[0]["value"]]
        values.extend(
            sum(pools[c["root_slot"], s["position"]][i] for i in indices) / len(indices)
            for s in states[1:-1]
        )
        values.append(states[-1]["value"])
        result.append(
            dict(token_advantages=token_credit([s["position"] for s in states], values))
        )
    return result


def credit_digest(rows, credits):
    payload = [
        (r["question_id"], r["root_index"], c["token_advantages"])
        for r, c in zip(rows, credits, strict=True)
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def question_contrast(rows, saved, pools, qid, offset, contrast):
    """Covariance-scaled MC16 contrast; endpoints zero, full-batch denominator."""
    assert offset in (0, 16) and contrast in range(3)
    result = []
    for row, credit in zip(rows, saved, strict=True):
        values = [0.0]
        for state in credit["states"][1:-1]:
            bits = pools[credit["root_slot"], state["position"]]
            value = 0.0
            if row["question_id"] == qid:
                value = (
                    sum(
                        sign * sum(bits[offset + 4 * k : offset + 4 * k + 4])
                        for k, sign in enumerate(WALSH[contrast])
                    )
                    / 16
                )
            values.append(value)
        values.append(0.0)
        result.append(
            dict(
                token_advantages=token_credit(
                    [s["position"] for s in credit["states"]], values
                )
            )
        )
    return result


def risk(gradient, contrasts, references):
    """Projected covariance of the normalized update at the pilot mean."""
    gg = gradient["norm2"]
    assert gg > 0 and len(contrasts) == 3
    values = [
        (
            DELTA
            / math.sqrt(gg)
            * (z["dots"][h] - gradient["dots"][h] * z["dots"]["pilot"] / gg)
        )
        ** 2
        for z in contrasts
        for h in references
    ]
    assert all(math.isfinite(v) for v in values)
    return math.fsum(values) / len(values)


def allocate(scores, counts):
    """One declared MC-count budget; no fitted quality cutoff or audit input."""
    assert set(scores) == set(counts) and all(c > 0 for c in counts.values())
    assert all(math.isfinite(s) and s >= 0 for s in scores.values())
    ceiling = 24 * sum(counts.values())
    used = 16 * sum(counts.values())
    budgets = dict.fromkeys(scores, 16)
    order = sorted(scores, key=lambda q: (-scores[q] / counts[q], q))
    for q in order:
        extra = 16 * counts[q]
        if used + extra <= ceiling:
            budgets[q] = 32
            used += extra
    return dict(
        budgets=budgets,
        order=order,
        calls=used,
        ceiling=ceiling,
        unused_calls=ceiling - used,
    )


def mixed_credits(rows, saved, pools, budgets):
    low = subset_credits(saved, pools, list(range(16)))
    full = subset_credits(saved, pools, list(range(32)))
    assert set(budgets) == {r["question_id"] for r in rows}
    assert set(budgets.values()) <= {16, 32}
    return [
        a if budgets[r["question_id"]] == 16 else b
        for r, a, b in zip(rows, low, full, strict=True)
    ]
