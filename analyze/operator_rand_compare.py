"""Overlay mean P(5)-vs-L curves from multiple operator-randomization runs.

Use to compare runs that differ in one factor (e.g. shuffle_seed) on a
single axis: if the curves coincide, that factor doesn't matter; if they
diverge, it does. The bare-control reference line is drawn if any run
carries a 'control' key (or one is supplied via --control).

Usage:
  py analyze/operator_rand_compare.py runs/seed0.json runs/seed1.json \
      --labels "seed 0" "seed 1" --control runs/operator_rand_control.json
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


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", nargs="+", help="one or more run JSONs")
    ap.add_argument("--labels", nargs="*", help="legend label per run")
    ap.add_argument("--control", help="JSON with a 'control' key, if not in a run")
    ap.add_argument("--out", default="runs/operator_rand_compare.png")
    args = ap.parse_args()

    runs = [load(p) for p in args.results_json]
    labels = args.labels or [Path(p).stem for p in args.results_json]

    control_p5 = None
    for r in runs:
        if r.get("control") is not None:
            control_p5 = r["control"]["p5"]
            break
    if control_p5 is None and args.control:
        cr = load(args.control)
        control_p5 = cr.get("control", {}).get("p5")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r, label, color in zip(runs, labels, [f"C{i}" for i in range(len(runs))]):
        Ls = sorted(int(k) for k in r["results"].keys())
        means = [r["results"][str(L)]["mean_p5"] for L in Ls]
        stds = [r["results"][str(L)]["std_p5"] for L in Ls]
        ax.errorbar(Ls, means, yerr=stds, marker="o", capsize=3,
                    color=color, label=label)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    if control_p5 is not None:
        ax.axhline(control_p5, color="C3", linestyle="-", linewidth=1.2,
                   label=f"control (no list) = {control_p5:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel(r"list length $L$ (log scale)")
    ax.set_ylabel(r"$P(5 \mid \{5,7\})$")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.set_title(f"{runs[0]['model']} — mean P(5) vs L across runs")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
