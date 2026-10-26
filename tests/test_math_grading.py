import json
import sys
from types import SimpleNamespace

import pytest

from rllab.math_grading import GradingTimeoutError, GradingWorker


@pytest.mark.parametrize("failure", ["timeout", "crash"])
def test_checker_timeout_or_crash_kills_child_without_a_verdict(failure):
    ending = "time.sleep(60)" if failure == "timeout" else "sys.exit(17)"
    program = (
        "import json,sys,time\n"
        "print(json.dumps({'ready':True}),flush=True)\n"
        "sys.stdin.readline()\n" + ending
    )
    error = GradingTimeoutError if failure == "timeout" else RuntimeError
    with GradingWorker(
        timeout=0.05, command=[sys.executable, "-u", "-c", program]
    ) as worker:
        with pytest.raises(error):
            worker.grade(r"\boxed{1}", "1")
        assert worker.process.poll() is not None


def test_failed_grading_preserves_raw_outputs_and_prior_verdicts(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from rllab import real_math_loop as loop

    events = []
    calls = 0

    def grade(text, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise GradingTimeoutError("synthetic stalled verifier")
        return dict(answer="1", reward=1, grading_status="correct")

    monkeypatch.setattr(
        loop, "GradingWorker", lambda: nullcontext(SimpleNamespace(grade=grade))
    )

    class Client:
        async def generate(self, **kw):
            return SimpleNamespace(
                token_ids=[1, 2],
                log_probs=None,
                extra_fields={"global_steps": 0},
                stop_reason="eos",
                num_preempted=0,
            )

    sampler = loop.NativeMathSampler(
        SimpleNamespace(decode=lambda *a, **k: r"\boxed{1}"),
        [dict(question_id="q", answer="1")],
        0.6,
        [2],
        tmp_path,
        lambda stage, **fields: events.append(dict(stage=stage, **fields)),
    )
    jobs = [
        dict(
            role="eval",
            question_id="q",
            root_slot=i,
            root_index=i,
            sample_index=i,
            position=0,
            prompt_ids=[3],
            response_prefix=[],
            seed=i,
        )
        for i in range(2)
    ]
    with pytest.raises(GradingTimeoutError):
        sampler.sample(Client(), jobs, version=0, phase="eval", concurrency=2)
    assert len((tmp_path / "eval.jsonl").read_text().splitlines()) == 2
    grades = [
        json.loads(x) for x in (tmp_path / "eval-grades.jsonl").read_text().splitlines()
    ]
    assert len(grades) == 1 and grades[0]["reward"] == 1
    assert events[-1]["stage"] == "grading_failed"
    assert events[-1]["index"] == events[-1]["sample_index"] == 1
    assert events[-1]["question_id"] == "q"
    assert "sampling_complete" not in [e["stage"] for e in events]


@pytest.mark.real_rl
def test_native_checker_nonintegral_factorial_and_original_equivalence(real_rl_stack):
    # This exact saved answer previously stalled general SymPy simplification.
    with GradingWorker(timeout=10) as worker:
        wrong = worker.grade(
            r"\boxed{\frac{2005^2}{2} \cdot \frac{1}{(2004 + \sqrt{2})!}}",
            r"2004 \sqrt{2} - 2003",
        )
        assert wrong == dict(
            answer=r"\frac{2005^2}{2} \cdot \frac{1}{(2004 + \sqrt{2})!}",
            reward=0,
            grading_status="extracted_wrong",
        )
        assert worker.grade(r"\boxed{\sqrt{8}/2}", r"\sqrt{2}")["reward"] == 1
        assert worker.grade(r"\boxed{(x+1)^2}", "x^2+2*x+1")["reward"] == 1
        assert worker.grade(r"\boxed{\frac{1}{2}}", "0.5")["reward"] == 1
        assert worker.grade(r"\boxed{\frac{2}{4}}", r"\frac{1}{2}")["reward"] == 0
        assert (
            worker.grade("unboxed answer 1", "1")["grading_status"]
            == "missing_or_incomplete_box"
        )
