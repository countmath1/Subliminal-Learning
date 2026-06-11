"""Aggregate many shuffle-seed runs of the operator-randomization experiment.

Reads <parent>/seed_*/operator_rand_results.json and produces:
  (top)    one violin per L showing the distribution, ACROSS seeds, of the
           per-seed mean P(5). Width = how content (different animals at each
           L per seed) spreads the mean. Control line overlaid.
  (bottom) "best-L" tally: for each seed, the L with the lowest mean P(5)
           (strongest pull toward 7); bars count how often each L wins.

Usage:
  py analyze/operator_rand_seedviolin.py runs/<array_parent_dir>
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import setup_style
setup_style()

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dir", help="dir containing seed_*/operator_rand_results.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.parent_dir, "seed_*",
                                          "operator_rand_results.json")))
    if not paths:
        raise SystemExit(f"no seed_*/operator_rand_results.json under {args.parent_dir}")

    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    model = runs[0]["model"]
    control_p5 = next((r["control"]["p5"] for r in runs if r.get("control")), None)

    # L -> list of per-seed means; also track each seed's argmin L.
    per_L = {}
    best_L = []
    for r in runs:
        Ls = sorted(int(k) for k in r["results"].keys())
        means = {L: r["results"][str(L)]["mean_p5"] for L in Ls}
        for L, m in means.items():
            per_L.setdefault(L, []).append(m)
        best_L.append(min(means, key=means.get))

    Ls = sorted(per_L.keys())
    data = [np.array(per_L[L]) for L in Ls]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5))

    # --- top: across-seed distribution of per-seed means, per L ---
    positions = np.arange(len(Ls))
    parts = ax1.violinplot(data, positions=positions, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("C0")
        pc.set_alpha(0.35)
    bp = ax1.boxplot(data, positions=positions, widths=0.25, showfliers=False,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_alpha(0.9)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(L) for L in Ls])
    ax1.set_xlabel(r"list length $L$")
    ax1.set_ylabel(r"per-seed mean $P(5 \mid \{5,7\})$")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    if control_p5 is not None:
        ax1.axhline(control_p5, color="C3", linewidth=1.2,
                    label=f"control (no list) = {control_p5:.3f}")
        ax1.legend(loc="best")
    ax1.set_title(f"Mean P(5) across {len(runs)} shuffle seeds, per L")

    # --- bottom: best-L tally ---
    counts = Counter(best_L)
    heights = [counts.get(L, 0) for L in Ls]
    ax2.bar(positions, heights, color="C0")
    ax2.set_xticks(positions)
    ax2.set_xticklabels([str(L) for L in Ls])
    ax2.set_xlabel(r"list length $L$")
    ax2.set_ylabel("# seeds where L is best")
    ax2.set_title(r"Most common 'best' $L$ (lowest mean $P(5)$ = strongest pull to 7)")
    for x, h in zip(positions, heights):
        if h:
            ax2.text(x, h + 0.1, str(h), ha="center", va="bottom", fontsize=9)

    fig.suptitle(model)
    plt.tight_layout()
    out = Path(args.out) if args.out else Path(args.parent_dir) / "seed_violin.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}  ({len(runs)} seeds)")


if __name__ == "__main__":
    main()
