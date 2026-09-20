#!/usr/bin/env python3
"""Turn results.jsonl into the winner table and the figures in figs/.

Two rules this script enforces, both learned the hard way:

  1. Winners are computed PER NETWORK. TP8/PP1 has no data at all on
     a100_pairwise_nvlink -- all 30 cells die with "Training data for model
     all_reduce is empty", because an 8-way all-reduce needs the NVSwitch
     fabric and pairwise NVLink only wires GPUs in twos. Pooling the two
     networks silently compares a 4-way field against a 3-way one.

  2. prefill=16384 is dropped. It sits past the fitted range of Vidur's
     attention profile for this model (CAVEATS.md #1), so those cells are
     extrapolated, not simulated.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
FIGS = ROOT / "figs"
IN_RANGE = [512, 2048, 8192]          # 16384 excluded, see CAVEATS.md #1
CONFIGS = [(1, 8), (2, 4), (4, 2), (8, 1)]
COLOUR = {(1, 8): "#b45309", (2, 4): "#0369a1", (4, 2): "#15803d", (8, 1): "#b91c1c"}
NETNAME = {"a100_dgx": "NVSwitch (DGX)", "a100_pairwise_nvlink": "pairwise NVLink"}
PDNAME = {32: "summarisation  (prefill:decode 32)",
          4: "chat  (4)",
          0.25: "long generation  (0.25)"}


def load():
    rows = [json.loads(l) for l in (ROOT / "results.jsonl").open()]
    return [r for r in rows if not r.get("error") and r["prefill"] in IN_RANGE]


def winners(rows, metric="e2e_p50"):
    """Best (tp, pp) for each (net, prefill, pd, qps) workload point."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["net"], r["prefill"], r["pd"], r["qps"])].append(r)
    out = {}
    for k, v in cells.items():
        b = min(v, key=lambda r: r[metric])
        out[k] = ((b["tp"], b["pp"]), b[metric], len(v))
    return out


def table(rows):
    w = winners(rows)
    lines = []
    for net in sorted({k[0] for k in w}):
        tally = defaultdict(int)
        for k, (cfg, _, _) in w.items():
            if k[0] == net:
                tally[cfg] += 1
        n = sum(tally.values())
        lines.append(f"\n{NETNAME[net]}  ({n} workload points)")
        for cfg, c in sorted(tally.items(), key=lambda x: -x[1]):
            lines.append(f"    TP{cfg[0]}/PP{cfg[1]}  wins {c:>2}/{n}")
        lines.append("    " + "-" * 52)
        lines.append(f"    {'prefill':>7} {'pd':>5} {'qps':>4}   winner      e2e p50 (s)   field")
        for k in sorted((k for k in w if k[0] == net), key=lambda k: (k[1], -k[2], k[3])):
            cfg, val, field = w[k]
            lines.append(f"    {k[1]:>7} {k[2]:>5} {k[3]:>4}   TP{cfg[0]}/PP{cfg[1]:<7}{val:>10.2f}{field:>8}")
    return "\n".join(lines)


def fig_crossover(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pds = [32, 4, 0.25]
    nets = ["a100_dgx", "a100_pairwise_nvlink"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
    for ri, net in enumerate(nets):
        for ci, pd in enumerate(pds):
            ax = axes[ri][ci]
            for cfg in CONFIGS:
                xs, ys = [], []
                for p in IN_RANGE:
                    v = [r["e2e_p50"] for r in rows
                         if r["net"] == net and r["pd"] == pd and r["prefill"] == p
                         and (r["tp"], r["pp"]) == cfg and r["qps"] == 1]
                    if v:
                        xs.append(p); ys.append(sum(v) / len(v))
                if xs:
                    ax.plot(xs, ys, "o-", color=COLOUR[cfg], lw=2, ms=5,
                            label=f"TP{cfg[0]}/PP{cfg[1]}")
            ax.set_xscale("log", base=2); ax.set_yscale("log")
            ax.set_xticks(IN_RANGE); ax.set_xticklabels([str(p) for p in IN_RANGE])
            ax.grid(alpha=.25, which="both", lw=.6)
            if ri == 0:
                ax.set_title(PDNAME[pd], fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{NETNAME[net]}\ne2e p50 (s), log", fontsize=9)
            if ri == 1:
                ax.set_xlabel("prefill tokens")
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Where the best parallelism split flips, at 1 qps  ·  Llama-3-70B, 8 A100s, Vidur",
                 fontsize=12)
    fig.text(.5, .015, "TP8/PP1 is absent from the bottom row: an 8-way all-reduce has no "
                       "pairwise-NVLink profile, so those 30 cells cannot run at all.",
             ha="center", fontsize=8.5, color="#475467")
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(FIGS / "crossover.png", dpi=150)
    plt.close(fig)


def fig_winner_map(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    w = winners(rows)
    nets = ["a100_dgx", "a100_pairwise_nvlink"]
    pds = [32, 4, 0.25]
    qpss = [1, 2, 4]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, net in zip(axes, nets):
        ys = [(pd, q) for pd in pds for q in qpss]
        for yi, (pd, q) in enumerate(ys):
            for xi, p in enumerate(IN_RANGE):
                hit = w.get((net, p, pd, q))
                if not hit:
                    ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor="#f1f5f9",
                                               edgecolor="w", lw=2))
                    continue
                cfg = hit[0]
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=COLOUR[cfg],
                                           edgecolor="w", lw=2, alpha=.85))
                ax.text(xi + .5, yi + .5, f"TP{cfg[0]}/PP{cfg[1]}", ha="center",
                        va="center", color="w", fontsize=8.5, weight="bold")
        ax.set_xlim(0, len(IN_RANGE)); ax.set_ylim(0, len(ys))
        ax.set_xticks([i + .5 for i in range(len(IN_RANGE))])
        ax.set_xticklabels([str(p) for p in IN_RANGE])
        ax.set_yticks([i + .5 for i in range(len(ys))])
        ax.set_yticklabels([f"pd {pd}, {q} qps" for pd, q in ys], fontsize=8)
        ax.set_xlabel("prefill tokens")
        ax.set_title(NETNAME[net], fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[Patch(facecolor=COLOUR[c], label=f"TP{c[0]}/PP{c[1]}") for c in CONFIGS],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Which split wins each workload  ·  lowest end-to-end p50", fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(FIGS / "winner-map.png", dpi=150)
    plt.close(fig)


def fig_topology_cost(rows):
    """What pairwise NVLink costs, for the configs that can run on both."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = .26
    cfgs = [(1, 8), (2, 4), (4, 2)]          # TP8 cannot run on pairwise at all
    for i, cfg in enumerate(cfgs):
        xs, ys = [], []
        for j, p in enumerate(IN_RANGE):
            d = [r["e2e_p50"] for r in rows if r["net"] == "a100_dgx"
                 and (r["tp"], r["pp"]) == cfg and r["prefill"] == p]
            n = [r["e2e_p50"] for r in rows if r["net"] == "a100_pairwise_nvlink"
                 and (r["tp"], r["pp"]) == cfg and r["prefill"] == p]
            if d and n:
                xs.append(j + (i - 1) * width)
                ys.append((sum(n) / len(n)) / (sum(d) / len(d)))
        ax.bar(xs, ys, width, color=COLOUR[cfg], label=f"TP{cfg[0]}/PP{cfg[1]}", alpha=.88)
    ax.axhline(1, color="#475467", lw=1, ls="--")
    ax.set_xticks(range(len(IN_RANGE)))
    ax.set_xticklabels([str(p) for p in IN_RANGE])
    ax.set_xlabel("prefill tokens")
    ax.set_ylabel("pairwise NVLink ÷ NVSwitch\n(e2e p50, >1 = slower)")
    ax.set_title("What the thinner interconnect costs, by how much tensor-parallelism you use",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "topology-cost.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    rows = load()
    print(f"{len(rows)} in-range cells (prefill 16384 excluded)")
    print(table(rows))
    fig_crossover(rows)
    fig_winner_map(rows)
    fig_topology_cost(rows)
    print(f"\nfigures -> {FIGS}")


if __name__ == "__main__":
    main()
