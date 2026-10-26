"""Isolate symbolic answer checking from the accelerator-owning driver."""

import json
import select
import subprocess
import sys


class GradingTimeoutError(TimeoutError):
    """No grading verdict was obtained; never interpret this as a wrong answer."""


class GradingWorker:
    """One lightweight CPU process per phase, with a deadline for every verdict.

    A thread or Python exception timer cannot bound third-party native work or
    code that catches BaseException. A child also leaves the training driver's
    SIGALRM/SIGTERM handlers intact and contains a symbolic-library crash.
    """

    def __init__(self, *, timeout=10.0, startup_timeout=60.0, command=None):
        assert timeout > 0 and startup_timeout > 0
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.command = command or [sys.executable, "-u", "-m", __name__]
        self.process = None

    def _read(self, seconds):
        if not select.select([self.process.stdout], [], [], seconds)[0]:
            raise GradingTimeoutError(f"Answer checker made no reply within {seconds}s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Answer checker exited without a verdict: {self.process.poll()}"
            )
        return json.loads(line)

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert self._read(self.startup_timeout) == {"ready": True}
        except BaseException:
            self.close()
            raise
        return self

    def grade(self, text, target):
        try:
            self.process.stdin.write(json.dumps([text, target]) + "\n")
            self.process.stdin.flush()
            return self._read(self.timeout)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()

    def __exit__(self, *_):
        self.close()


def main():
    from rllab._vendor.prm800k import grader  # warm imports before ready
    from rllab.math_credit import grade_completion

    assert callable(grader.grade_answer)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        text, target = json.loads(line)
        print(json.dumps(grade_completion(text, target)), flush=True)


if __name__ == "__main__":
    main()
