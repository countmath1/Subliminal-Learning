"""Plot a chosen subset of conditions from an exp1 results JSON.

Same axes and styling as `exp1_plot.py`, but you pick which conditions to
include (and their left-to-right order) on the command line. Useful for
focused comparisons when the full run has more conditions than you want
in a single figure.

Usage:
    py analyze/exp1_plot_subset.py runs/<file>.json control_reversed treatment

Output PNG is written next to the input JSON, with the chosen condition
names appended to the stem so multiple subset plots can coexist.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("conditions", nargs="+",
                    help="Condition names to plot, in left-to-right order")
    args = ap.parse_args()

    with open(args.results_json) as f:
        r = json.load(f)

    names = args.conditions
    missing = [n for n in names if n not in r["conditions"]]
    if missing:
        avail = list(r["conditions"].keys())
        raise SystemExit(f"conditions not in JSON: {missing}; available: {avail}")

    p_logit = [r["conditions"][n]["logit"]["p_5_given_5_or_7"] for n in names]

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names) + 2), 4))
    x = np.arange(len(names))
    colors = ["grey" if n.startswith("control") else "C0" for n in names]
    ax.bar(x, p_logit, color=colors)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xticks(x)
    rotate = len(names) > 3
    ax.set_xticklabels(names,
                       rotation=20 if rotate else 0,
                       ha="right" if rotate else "center")

    ax.set_ylabel(r"$P(5 \mid \mathrm{answer} \in \{5, 7\})$")
    ax.set_ylim(0, 1)
    ax.set_title(r["model"])

    for xi, pi in zip(x, p_logit):
        ax.text(xi, pi + 0.02, f"{pi:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    src = Path(args.results_json)
    out = src.parent / (src.stem + "_" + "_".join(names) + ".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
