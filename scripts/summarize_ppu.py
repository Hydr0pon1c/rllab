#!/usr/bin/env python3
"""Summarise per-card utilisation logs, including exact two-hour windows."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_ppu.py LOG.csv", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    by_device: dict[int, list[tuple[datetime, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.strptime(
                row["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            device = int(row["index"])
            by_device.setdefault(device, []).append(
                (timestamp, float(row["utilization_percent"]))
            )
    report = {"path": str(path.resolve()), "devices": {}}
    for device, samples in sorted(by_device.items()):
        samples.sort()
        values = [value for _, value in samples]
        window_means: list[float] = []
        left = 0
        running = 0.0
        for right, (timestamp, value) in enumerate(samples):
            running += value
            while timestamp - samples[left][0] > timedelta(hours=2):
                running -= samples[left][1]
                left += 1
            if timestamp - samples[left][0] >= timedelta(minutes=119):
                window_means.append(running / (right - left + 1))
        report["devices"][str(device)] = {
            "samples": len(samples),
            "mean_utilization_percent": sum(values) / len(values) if values else None,
            "minimum_two_hour_mean_percent": min(window_means)
            if window_means
            else None,
            "two_hour_windows_observed": len(window_means),
        }
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
