"""Current-policy math sampling and supplied block credit for the native trainer."""

import asyncio
from collections import defaultdict
from contextlib import nullcontext
import json
from statistics import mean
import time

from rllab.math_credit import (
    extract_answer,
    keyed_seed,
    process_token_credit,
    value_summary,
)
from rllab.math_grading import GradingWorker

BOUNDARIES = [32, 64, 128, 256, 512, 1024, 2048]
TOKEN_CAP = 3072
SEED = 20260914


def root_jobs(
    questions, prompts, actor, version, roots_per_question, *, seed=SEED, role="root"
):
    return [
        dict(
            role=role,
            question_id=q["question_id"],
            root_slot=qi * roots_per_question + i,
            root_index=i,
            sample_index=i,
            position=0,
            prompt_ids=prompts[q["question_id"]],
            response_prefix=[],
            seed=keyed_seed(
                seed, "native-math", actor, version, q["question_id"], role, i
            ),
        )
        for qi, q in enumerate(questions)
        for i in range(roots_per_question)
    ]


def mc_prefix_counts(roots, count, question_ids=None, prefix_counts=None):
    """Validate complete prefix coverage before any MC sampling or assembly."""
    if question_ids is not None:
        assert set(question_ids) <= {r["question_id"] for r in roots}
    expected = {
        (i, p): count
        for i, r in enumerate(roots)
        for p in BOUNDARIES
        if question_ids is None or r["question_id"] in question_ids
        if p < len(r["generated_ids"])
    }
    counts = expected if prefix_counts is None else prefix_counts
    assert set(counts) == set(expected)
    assert all(type(n) is int and n > 0 and n % 2 == 0 for n in counts.values())
    return counts


def suffix_jobs(
    roots,
    prompts,
    actor,
    version,
    count,
    *,
    namespace="suffix",
    seed=SEED,
    question_ids=None,
    prefix_counts=None,
):
    # Interleave prefix pools so a bounded packet covers the root batch.
    counts = mc_prefix_counts(roots, count, question_ids, prefix_counts)
    return [
        dict(
            role=namespace,
            question_id=r["question_id"],
            root_slot=slot,
            root_index=r["root_index"],
            sample_index=i,
            position=p,
            prompt_ids=prompts[r["question_id"]] + r["generated_ids"][:p],
            response_prefix=r["generated_ids"][:p],
            seed=keyed_seed(
                seed,
                "native-math",
                actor,
                version,
                r["question_id"],
                namespace,
                r["root_index"],
                p,
                i,
            ),
        )
        for i in range(max(counts.values(), default=0))
        for slot, r in enumerate(roots)
        for p in BOUNDARIES
        if i < counts.get((slot, p), 0)
    ]


def assemble_credits(roots, suffixes, count, *, question_ids=None, prefix_counts=None):
    # Empty prefix lists give ordinary terminal LOO credit to the other groups.
    counts = mc_prefix_counts(roots, count, question_ids, prefix_counts)
    pools = defaultdict(dict)
    for row in suffixes:
        key = row["root_slot"], row["position"]
        assert row["sample_index"] not in pools[key]
        pools[key][row["sample_index"]] = row["reward"]
    assert set(pools) == set(counts)
    credits = []
    for i, r in enumerate(roots):
        peers = [
            s["reward"]
            for j, s in enumerate(roots)
            if j != i and s["question_id"] == r["question_id"]
        ]
        assert peers
        prefixes = []
        for p in BOUNDARIES:
            if (i, p) in pools:
                pool = pools[i, p]
                n = counts[i, p]
                assert set(pool) == set(range(n))
                prefixes.append(
                    dict(position=p, **value_summary([pool[j] for j in range(n)]))
                )
        credits.append(
            dict(
                question_id=r["question_id"],
                root_slot=i,
                root_index=r["root_index"],
                **process_token_credit(
                    len(r["generated_ids"]), r["reward"], mean(peers), prefixes
                ),
            )
        )
    return credits


class NativeMathSampler:
    """Time native requests separately from decoding, grading and artifact writes."""

    def __init__(self, tokenizer, questions, temperature, eos, folder, event):
        self.tokenizer = tokenizer
        self.questions = {q["question_id"]: q for q in questions}
        self.temperature = temperature
        self.eos = eos
        self.folder = folder
        self.event = event
        self.total_tokens = 0

    def sample(self, client, jobs, *, version, phase, concurrency, grade=True):
        outputs = [None] * len(jobs)
        started = time.monotonic()
        self.event(
            "sampling_started",
            phase=phase,
            requests=len(jobs),
            concurrency=concurrency,
            version=version,
        )

        async def run():
            semaphore = asyncio.Semaphore(concurrency)
            completed, last_progress = 0, time.monotonic()

            async def one(index, job):
                nonlocal completed, last_progress
                async with semaphore:
                    output = await client.generate(
                        request_id=f"{phase}-{index}",
                        prompt_ids=job["prompt_ids"],
                        sampling_params=dict(
                            max_tokens=TOKEN_CAP - job["position"],
                            temperature=self.temperature,
                            top_p=1.0,
                            top_k=-1,
                            repetition_penalty=1.0,
                            stop_token_ids=self.eos,
                            ignore_eos=False,
                            seed=job["seed"],
                            logprobs=job.get("return_log_probs", job["role"] == "root"),
                        ),
                    )
                    outputs[index] = output
                    assert output.extra_fields["global_steps"] == version
                    ids = output.token_ids
                    assert 0 < len(ids) <= TOKEN_CAP - job["position"]
                    assert not any(t in self.eos for t in ids[:-1])
                    assert (
                        ids[-1] in self.eos or len(ids) == TOKEN_CAP - job["position"]
                    )
                    completed += 1
                    if time.monotonic() - last_progress >= 25:
                        self.event(
                            "sampling_progress",
                            phase=phase,
                            completed=completed,
                            requests=len(jobs),
                        )
                        last_progress = time.monotonic()

            await asyncio.gather(*(one(i, job) for i, job in enumerate(jobs)))

        try:
            asyncio.run(run())
        finally:
            generation_seconds = time.monotonic() - started
            # Persist completed outputs even when a request or invariant fails.
            written = time.monotonic()
            records = []
            with (self.folder / f"{phase}.jsonl").open("x") as stream:
                for job, output in zip(jobs, outputs, strict=True):
                    if output is None:
                        continue
                    record = {
                        **{
                            k: v
                            for k, v in job.items()
                            if k not in ("prompt_ids", "response_prefix")
                        },
                        "generated_ids": output.token_ids,
                        "log_probs": output.log_probs,
                        "stop_reason": output.stop_reason,
                        "num_preempted": output.num_preempted,
                        "extra_fields": output.extra_fields,
                        "weight_version": version,
                        "remaining_budget": TOKEN_CAP - job["position"],
                        "hit_token_cap": output.token_ids[-1] not in self.eos
                        if output.token_ids
                        else False,
                    }
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
            io_seconds = time.monotonic() - written
        self.total_tokens += sum(len(r["generated_ids"]) for r in records)
        self.event(
            "generation_complete",
            phase=phase,
            version=version,
            requests=len(records),
            generation_seconds=generation_seconds,
            io_seconds=io_seconds,
        )
        before = time.monotonic()
        if grade:
            self.event("grading_started", phase=phase, requests=len(records))
            last_progress = before
            context = GradingWorker() if records else nullcontext()
            with (
                (self.folder / f"{phase}-grades.jsonl").open("x") as stream,
                context as checker,
            ):
                for index, (job, record) in enumerate(zip(jobs, records, strict=True)):
                    response = self.tokenizer.decode(
                        job["response_prefix"] + record["generated_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    target = self.questions[job["question_id"]]["answer"]
                    try:
                        verdict = checker.grade(response, target)
                        assert verdict["grading_status"] != "grader_error", verdict
                    except BaseException as exc:
                        self.event(
                            "grading_failed",
                            phase=phase,
                            index=index,
                            question_id=job["question_id"],
                            root_slot=job["root_slot"],
                            position=job["position"],
                            sample_index=job["sample_index"],
                            answer=extract_answer(response),
                            target=target,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        raise
                    record.update(verdict)
                    stream.write(
                        json.dumps(
                            {
                                k: record[k]
                                for k in (
                                    "root_slot",
                                    "position",
                                    "sample_index",
                                    "reward",
                                    "answer",
                                    "grading_status",
                                )
                            }
                        )
                        + "\n"
                    )
                    if time.monotonic() - last_progress >= 25:
                        stream.flush()
                        self.event(
                            "grading_progress",
                            phase=phase,
                            completed=index + 1,
                            requests=len(records),
                        )
                        last_progress = time.monotonic()
        cpu_seconds = time.monotonic() - before
        tokens = sum(len(r["generated_ids"]) for r in records)
        self.event(
            "sampling_complete",
            phase=phase,
            version=version,
            concurrency=concurrency,
            requests=len(records),
            tokens=tokens,
            generation_seconds=generation_seconds,
            io_seconds=io_seconds,
            cpu_seconds=cpu_seconds,
            tokens_per_second=tokens / generation_seconds,
            cap_count=sum(r["hit_token_cap"] for r in records),
            correct=sum(r["reward"] for r in records) if grade else None,
        )
        return records
