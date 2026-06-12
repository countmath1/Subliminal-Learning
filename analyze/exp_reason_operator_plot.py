"""Plot thinking-mode P(5|{5,7}) vs L, with Wilson binomial CIs.

Each L contributes one proportion (from n_iters binary answers). Error bars
are 95% Wilson intervals. Optionally overlay the logit (non-thinking)
mean-vs-L curve via --logit, to compare reasoning vs no-reasoning.

Usage:
  py analyze/exp_reason_operator_plot.py runs/<thinking>.json
  py analyze/exp_reason_operator_plot.py runs/<thinking>.json --logit runs/<logit_sweep>.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup_style
setup_style()

import matplotlib.pyplot as plt
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, center - half, center + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--logit", help="logit-sweep results JSON to overlay")
    args = ap.parse_args()

    with open(args.results_json) as f:
        r = json.load(f)

    Ls = sorted(int(k) for k in r["results"].keys())

    def series(count_key):
        xs, p, lo, hi = [], [], [], []
        for L in Ls:
            c = r["results"][str(L)].get(count_key)
            if not c:
                continue
            n = c["5"] + c["7"]
            if n == 0:
                continue
            pp, l, h = wilson(c["5"], n)
            xs.append(L); p.append(pp); lo.append(pp - l); hi.append(h - pp)
        return xs, p, lo, hi

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # All traces, then the two splits: fully-reasoned (natural) is the clean
    # measurement; forced exposes any bias from cutting reasoning at the budget.
    for key, color, label in [
        ("counts", "C0", "thinking — all"),
        ("counts_natural", "C2", "thinking — fully reasoned"),
        ("counts_forced", "C4", "thinking — forced"),
    ]:
        xs, p, lo, hi = series(key)
        if xs:
            ax.errorbar(xs, p, yerr=[lo, hi], marker="o", capsize=3, color=color,
                        label=label, alpha=0.9 if key != "counts_forced" else 0.5)

    if args.logit:
        with open(args.logit) as f:
            lr = json.load(f)
        lLs = sorted(int(k) for k in lr["results"].keys())
        lmean = [lr["results"][str(L)]["mean_p5"] for L in lLs]
        ax.plot(lLs, lmean, marker="s", color="C1", alpha=0.8,
                label="non-thinking (logit mean)")
        ctrl = lr.get("control", {}).get("p5")
        if ctrl is not None:
            ax.axhline(ctrl, color="C3", linewidth=1.0,
                       label=f"control (no list) = {ctrl:.3f}")

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"list length $L$ (log scale)")
    ax.set_ylabel(r"$P(5 \mid \{5,7\})$")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.set_title(f"{r['model']} — thinking-mode P(5) vs L")
    plt.tight_layout()
    out = Path(args.results_json).with_suffix(".png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
