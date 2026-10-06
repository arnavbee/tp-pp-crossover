#!/usr/bin/env python3
"""Recompute every number the README claims, straight from results.jsonl.

No GPUs, no Vidur, no matplotlib: just the committed data. Run it after any
change to results.jsonl or to the analysis, and the README cannot drift away
from the sweep without this failing.

    python3 verify.py          # prints each claim, PASS/FAIL, exit 1 on any FAIL

The two rules analyse.py enforces apply here too: winners are computed PER
NETWORK, and prefill=16384 is excluded (CAVEATS.md #1).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
IN_RANGE = [512, 2048, 8192]          # 16384 excluded, see CAVEATS.md #1
ALL_REDUCE_ERR = "Exception: Training data for model all_reduce is empty"

fails = 0


def check(claim, ok, got):
    global fails
    if not ok:
        fails += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}\n         got: {got}")


def near(a, b, tol):
    return abs(a - b) <= tol


def winners(rows, metric="e2e_p50"):
    """Best (tp, pp) per (net, prefill, pd, qps) workload point."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["net"], r["prefill"], r["pd"], r["qps"])].append(r)
    return {k: (min(v, key=lambda r: r[metric])["tp"],
                min(v, key=lambda r: r[metric])["pp"]) for k, v in cells.items()}


def tally(w, net):
    t = defaultdict(int)
    for k, cfg in w.items():
        if k[0] == net:
            t[cfg] += 1
    return t


def ratio(rows, cfg, prefill):
    """pairwise NVLink / NVSwitch, mean e2e p50 over the cells that ran on both."""
    d = [r["e2e_p50"] for r in rows if r["net"] == "a100_dgx"
         and (r["tp"], r["pp"]) == cfg and r["prefill"] == prefill]
    n = [r["e2e_p50"] for r in rows if r["net"] == "a100_pairwise_nvlink"
         and (r["tp"], r["pp"]) == cfg and r["prefill"] == prefill]
    return (sum(n) / len(n)) / (sum(d) / len(d))


def main():
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").open()]
    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    inr = [r for r in ok if r["prefill"] in IN_RANGE]
    w = winners(inr)

    print("Cell counts")
    check("240 cells swept, 210 ran",
          len(rows) == 240 and len(ok) == 210, f"{len(rows)} swept, {len(ok)} ran")
    check("168 in-range cells reported",
          len(inr) == 168, len(inr))
    check("48 cells excluded at prefill 16384 (CAVEATS.md #1)",
          sum(1 for r in rows if r["prefill"] == 16384) == 48,
          sum(1 for r in rows if r["prefill"] == 16384))

    print("\nTP8/PP1 cannot run on pairwise NVLink")
    t8pw = [r for r in rows if (r["tp"], r["pp"]) == (8, 1)
            and r["net"] == "a100_pairwise_nvlink"]
    check("all 30 of its pairwise cells fail, and they are the only failures",
          len(t8pw) == 30 and all(r.get("error") for r in t8pw) and len(errs) == 30,
          f"{len(t8pw)} cells, {sum(1 for r in t8pw if r.get('error'))} failed, "
          f"{len(errs)} failures in the sweep")
    check(f"every failure is {ALL_REDUCE_ERR!r}",
          {r["error"] for r in errs} == {ALL_REDUCE_ERR},
          sorted({r["error"] for r in errs}))

    print("\nNVSwitch (DGX): TP8/PP1 wins, except at long context")
    td = tally(w, "a100_dgx")
    check("TP8/PP1 wins 16 of 24 workload points",
          td[(8, 1)] == 16 and sum(td.values()) == 24,
          f"TP8/PP1 {td[(8, 1)]}/{sum(td.values())}, TP4/PP2 {td[(4, 2)]}")
    at8192 = {cfg for k, cfg in w.items() if k[0] == "a100_dgx" and k[1] == 8192}
    check("TP4/PP2 takes every point at prefill 8192",
          at8192 == {(4, 2)}, sorted(at8192))
    at2048_4qps = {k[2]: cfg for k, cfg in w.items()
                   if k[0] == "a100_dgx" and k[1] == 2048 and k[3] == 4}
    check("at 2048 and 4 qps it already wins pd 32 and pd 4, but not pd 0.25",
          at2048_4qps.get(32) == (4, 2) and at2048_4qps.get(4) == (4, 2)
          and at2048_4qps.get(0.25) == (8, 1),
          {f"pd {pd}": f"TP{c[0]}/PP{c[1]}" for pd, c in sorted(at2048_4qps.items())})

    print("\nPairwise NVLink: the crossover shifts one rung")
    tp = tally(w, "a100_pairwise_nvlink")
    check("TP4/PP2 wins 15 of 24, TP2/PP4 the other 9",
          tp[(4, 2)] == 15 and tp[(2, 4)] == 9 and sum(tp.values()) == 24,
          f"TP4/PP2 {tp[(4, 2)]}, TP2/PP4 {tp[(2, 4)]}, of {sum(tp.values())}")
    pw8192 = {cfg for k, cfg in w.items()
              if k[0] == "a100_pairwise_nvlink" and k[1] == 8192}
    check("TP2/PP4 takes every point at prefill 8192",
          pw8192 == {(2, 4)}, sorted(pw8192))

    print("\nWhat the thinner fabric costs")
    for cfg in [(1, 8), (2, 4)]:
        rs = {p: ratio(inr, cfg, p) for p in IN_RANGE}
        check(f"TP{cfg[0]}/PP{cfg[1]} unchanged, within 3% at every prefill",
              all(r <= 1.03 for r in rs.values()),
              {p: f"{(r - 1) * 100:+.1f}%" for p, r in rs.items()})
    rs = {p: ratio(inr, (4, 2), p) for p in IN_RANGE}
    check("TP4/PP2 pays 27% at 512, 21% at 2048, 51% at 8192",
          near(rs[512], 1.27, .005) and near(rs[2048], 1.21, .005)
          and near(rs[8192], 1.51, .005),
          {p: f"{(r - 1) * 100:+.1f}%" for p, r in rs.items()})
    check("and the cost grows with the message it has to reduce",
          rs[2048] < rs[8192], f"2048 {rs[2048]:.3f} < 8192 {rs[8192]:.3f}")

    print(f"\n{fails} failed" if fails else "\nall claims hold")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
