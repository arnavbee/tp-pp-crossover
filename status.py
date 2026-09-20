#!/usr/bin/env python3
"""Honest status for the sweep. Counts ERRORS, not rows.

Written 2026-09-15 after reporting the sweep as "running fine" twice on row count
alone while 58% of its cells were dying with a bare KeyError. A row is not a result.
Exit code 1 if anything is wrong, so a cron can gate on it.
"""
import json, subprocess, sys, collections
from pathlib import Path

HERE = Path(__file__).parent
TOTAL = 240

def main():
    res = HERE / "results.jsonl"
    rows = [json.loads(l) for l in res.open()] if res.exists() else []
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]

    alive = subprocess.run(["pgrep", "-f", "sweep.py"], capture_output=True, text=True).stdout.split()
    vidur = subprocess.run(["pgrep", "-f", "vidur.main"], capture_output=True, text=True).stdout.split()

    bad = False
    print(f"cells: {len(ok)} good / {len(errs)} failed / {TOTAL} total"
          f"  ({100*len(rows)//TOTAL if TOTAL else 0}% attempted)")
    if errs:
        bad = True
        c = collections.Counter(e["error"][:70] for e in errs)
        print(f"!! {len(errs)} FAILED ({100*len(errs)//max(1,len(rows))}% of attempted):")
        for k, v in c.most_common(5):
            print(f"   {v:>4} x {k}")
    if ok:
        # a good row with no metrics is a silent failure too
        empty = [r for r in ok if not r.get("n")]
        if empty:
            bad = True
            print(f"!! {len(empty)} rows exited 0 but produced no request metrics")
        last = ok[-1]
        print(f"last good: {last['tag']}  e2e_p50={last.get('e2e_p50')} "
              f"tbt_p99={last.get('tbt_p99')} wall={last.get('wall_s')}s")

    # ⚠️ Vidur's a100/Llama-3-70B attention profile was measured at max_model_len 16384 and
    # kv_cache_size tops out at 16320. MAX_TOKENS in sweep.py is 32768, so any cell whose
    # prefill+decode exceeds 16320 is the random forest EXTRAPOLATING, not predicting. Those
    # rows run clean and mean nothing. mlp.csv goes to 32768, so only attention is out of range.
    PROFILED_KV = 16320
    def total(r):
        return r["prefill"] + max(1, round(r["prefill"] / r["pd"])) + 16
    oor = [r for r in ok if total(r) > PROFILED_KV]
    if oor:
        bad = True
        combos = sorted({(r["prefill"], r["pd"]) for r in oor})
        print(f"!! {len(oor)} rows are OUTSIDE the profiled attention range "
              f"(>{PROFILED_KV} kv tokens): prefill/pd {combos}")
        print("   these are extrapolated, not simulated. Exclude them or reprofile.")

    if not alive:
        print("!! sweep.py is NOT running")
        bad = True
    else:
        print(f"sweep alive (pid {alive[0]}), vidur workers: {len(vidur)}")

    # 8GB machine; the 32768 predictor fit is the memory risk, not the disk
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    d = {k.strip(): int(v.strip().rstrip('.')) for k, v in
         (l.split(':', 1) for l in vm.splitlines() if ':' in l and l.split(':')[1].strip().rstrip('.').isdigit())}
    free_gb = (d.get("Pages free", 0) + d.get("Pages inactive", 0)) * 16384 / 1e9
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    print(f"free+inactive: {free_gb:.1f}GB | {swap}")
    if free_gb < 0.6:
        print("!! memory pressure: the predictor fit may get killed")
        bad = True

    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
