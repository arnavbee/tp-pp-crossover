#!/usr/bin/env python3
"""Derive makespan for every finished run directory.

Why this is a separate script and not a fix inside sweep.py: sweep.py's collect()
reads a column named `request_completion_time`, which Vidur does not emit. The
guard `if rows[0].get(...) else None` meant makespan_s came out null on every
row and nobody noticed. sweep.py is mid-run as of 2026-09-15 23:57 and its loky
workers re-import the module, so it is not safe to edit live. This backfills the
metric from data already on disk.

Makespan = when the last request finished, measured from when the first arrived.
Vidur gives inter-arrival gaps, not absolute arrival times, so arrival_i is the
running sum of `request_inter_arrival_delay` (blank on the first row = t0) and
completion_i = arrival_i + request_e2e_time.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).parent


def makespan(csv_path):
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    t = 0.0
    last = None
    for r in rows:
        gap = r.get("request_inter_arrival_delay", "")
        t += float(gap) if gap not in ("", None) else 0.0
        e2e = r.get("request_e2e_time", "")
        if e2e in ("", None):
            continue
        last = t + float(e2e) if last is None else max(last, t + float(e2e))
    # None, not 0.0. 12 cells in the 15 Sep run have 128 rows and not one finished
    # request (pd=0.25 at prefill>=2048 generates past the simulated horizon). The old
    # `last = 0.0` start reported those as a 0-second makespan, which plots as the
    # fastest config on the board. An unfinished cell has no makespan.
    return None if last is None else round(last, 2)


def main():
    out = []
    for f in sorted(ROOT.glob("runs/*/*/request_metrics.csv")):
        n = sum(1 for _ in f.open()) - 1
        if n < 128:            # partial or still-running cell
            continue
        out.append({"dir": str(f.parent.relative_to(ROOT)),
                    "cell": f.parent.parent.name,
                    "n": n,
                    "makespan_s": makespan(f)})
    dest = ROOT / "makespan.jsonl"
    with dest.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out)} complete cells -> {dest}")
    if out:
        vals = [r["makespan_s"] for r in out if r["makespan_s"]]
        print(f"makespan range: {min(vals)}s .. {max(vals)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
