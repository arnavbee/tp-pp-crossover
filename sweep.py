#!/usr/bin/env python3
"""
Where does the best parallelism config flip from tensor-parallel to pipeline-parallel?

Sweeps (TP, PP) at a fixed world size across context length, prefill:decode ratio,
load, and interconnect topology, using Vidur (MLSys'24) as the execution engine.

Cost note: Vidur caches a random-forest execution-time predictor per
(model, device, tensor_parallel_size). The first run at each TP costs ~11 min on
8 cores; every later run at that TP is seconds. So the grid is cheap once the
four fits are paid for.
"""
import argparse, csv, itertools, json, os, subprocess, sys, time
from pathlib import Path

VIDUR = Path.home() / "code" / "vidur"
PY = VIDUR / ".venv" / "bin" / "python"
OUT = Path(__file__).parent / "runs"

MODEL = "meta-llama/Meta-Llama-3-70B"
DEVICE = "a100"

# world size 8: every way to split it between tensor- and pipeline-parallel
CONFIGS = [(1, 8), (2, 4), (4, 2), (8, 1)]
# the interconnect axis. dgx = fat all-to-all NVSwitch, pairwise = NVLink pairs only.
NETWORKS = ["a100_dgx", "a100_pairwise_nvlink"]
# context length (prefill tokens)
PREFILLS = [512, 2048, 8192, 16384]
# prefill:decode shape. 32 = summarisation, 4 = chat, 0.25 = agent/long-generation
PD_RATIOS = [32, 4, 0.25]
QPS = [1, 2, 4]

# Predictor range. Every cell must satisfy prefill + decode + 16 <= MAX_TOKENS or the
# run dies inside Vidur's lookup table. MAX_TOKENS is the single knob that decides both
# how long the four fits take and how much of the context axis survives.
MAX_TOKENS = 32768
MAX_CHUNK = 4096


def feasible(prefill, pd):
    """Cells the predictor can actually answer. See MAX_TOKENS."""
    decode = max(1, int(round(prefill / pd)))
    return prefill + decode + 16 <= MAX_TOKENS


def run_one(tp, pp, net, prefill, pd, qps, tag, dry=False):
    decode = max(1, int(round(prefill / pd)))
    maxtok = prefill + decode + 16
    outdir = OUT / tag
    cmd = [
        str(PY), "-m", "vidur.main",
        "--replica_config_model_name", MODEL,
        "--replica_config_device", DEVICE,
        "--replica_config_network_device", net,
        "--replica_config_tensor_parallel_size", str(tp),
        "--replica_config_num_pipeline_stages", str(pp),
        "--length_generator_config_type", "fixed",
        "--fixed_request_length_generator_config_prefill_tokens", str(prefill),
        "--fixed_request_length_generator_config_decode_tokens", str(decode),
        "--fixed_request_length_generator_config_max_tokens", str(maxtok),
        "--interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(qps),
        "--synthetic_request_generator_config_num_requests", "128",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--metrics_config_output_dir", str(outdir),
        # Vidur builds a lookup table of predicted times bounded by these. The defaults
        # (4096 tokens/request, 4096 chunk) are far below this grid, and every cell past
        # them dies with a bare KeyError on the (chunk, kv) pair. Raising them is what
        # makes long-context cells runnable at all; it also makes each predictor fit
        # markedly more expensive, which is the price of the context-length axis.
        "--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(MAX_TOKENS),
        "--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(MAX_CHUNK),
    ]
    if dry:
        print(" ".join(cmd)); return None
    t0 = time.time()
    p = subprocess.run(cmd, cwd=VIDUR, capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": p.stderr.strip().splitlines()[-1] if p.stderr else "failed"}
    # vidur writes into a timestamped subdir
    sub = sorted(outdir.glob("*/"), key=os.path.getmtime)[-1]
    return {"dir": str(sub), "wall_s": round(time.time() - t0, 1)}


def collect(sub):
    """Pull the numbers that decide the crossover out of one run."""
    d = Path(sub)
    f = d / "request_metrics.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open()))
    if not rows:
        return {}

    def col(name):
        vals = [float(r[name]) for r in rows if r.get(name) not in (None, "")]
        return sorted(vals)

    def pct(vals, p):
        if not vals: return None
        i = min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))
        return round(vals[i], 4)

    e2e = col("request_e2e_time")
    ttft = col("prefill_e2e_time") or col("request_scheduling_delay")
    sched = col("request_scheduling_delay")
    out = {
        "n": len(rows),
        "e2e_p50": pct(e2e, 50), "e2e_p99": pct(e2e, 99),
        "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "sched_delay_p99": pct(sched, 99),
    }
    # normalised decode time = per-output-token latency, the SLO people actually serve to
    tbt = col("decode_time_execution_plus_preemption_normalized")
    if tbt:
        out["tbt_p50"] = pct(tbt, 50); out["tbt_p99"] = pct(tbt, 99)
    # Makespan: when the last request finished, measured from when the first arrived.
    # Vidur emits inter-arrival GAPS, not absolute arrival times, and has no
    # `request_completion_time` column at all -- reading one is what made this key null
    # on all 240 rows of the first run (see CAVEATS.md #4). Derive it instead.
    if e2e:
        t = last = 0.0
        for r in rows:
            gap = r.get("request_inter_arrival_delay", "")
            t += float(gap) if gap not in ("", None) else 0.0
            v = r.get("request_e2e_time", "")
            if v not in ("", None):
                last = max(last, t + float(v))
        out["makespan_s"] = round(last, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--warm", action="store_true",
                    help="only pay the four predictor fits, one cheap run per TP")
    ap.add_argument("--networks", nargs="*", default=NETWORKS)
    ap.add_argument("--prefills", nargs="*", type=int, default=PREFILLS)
    ap.add_argument("--qps", nargs="*", type=float, default=QPS)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    results_path = Path(__file__).parent / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.open():
            r = json.loads(line)
            # only completed cells count as done. A failed cell must be retried on
            # resume, not skipped -- skipping is how a broken grid looks finished.
            if "error" not in r:
                done.add(r["tag"])

    if a.warm:
        grid = [(tp, pp, "a100_dgx", 512, 4, 1) for tp, pp in CONFIGS]
    else:
        grid = [(tp, pp, net, pre, pd, q)
                for (tp, pp) in CONFIGS
                for net in a.networks
                for pre in a.prefills
                for pd in PD_RATIOS
                for q in a.qps
                if feasible(pre, pd)]
        dropped = sorted({(pre, pd) for pre in a.prefills for pd in PD_RATIOS
                          if not feasible(pre, pd)})
        if dropped:
            print(f"dropped as out of predictor range (> {MAX_TOKENS} tokens): {dropped}")

    print(f"{len(grid)} cells, {len(done)} already done")
    with results_path.open("a") as fh:
        for i, (tp, pp, net, pre, pd, q) in enumerate(grid, 1):
            tag = f"tp{tp}_pp{pp}_{net}_pre{pre}_pd{pd}_qps{q}"
            if tag in done:
                continue
            print(f"[{i}/{len(grid)}] {tag}", flush=True)
            r = run_one(tp, pp, net, pre, pd, q, tag, dry=a.dry)
            if a.dry or r is None:
                continue
            rec = {"tag": tag, "tp": tp, "pp": pp, "net": net,
                   "prefill": pre, "pd": pd, "qps": q, **r}
            if "dir" in r:
                rec.update(collect(r["dir"]))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print("   ", {k: v for k, v in rec.items() if k in
                          ("wall_s", "e2e_p50", "ttft_p99", "tbt_p99", "error")}, flush=True)


if __name__ == "__main__":
    main()
