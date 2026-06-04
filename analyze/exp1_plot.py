"""Plot exp1 results: logit-based P(5 | answer in {5,7}) per condition.

Usage:  python analyze/exp1_plot.py runs/<jobid>/exp1_results.json

The plot scales to any number of conditions — add more keys to the
`conditions` dict in the results JSON (e.g. one per preference category)
and they'll each get their own bar, with "control" pinned first.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    args = ap.parse_args()

    with open(args.results_json) as f:
        r = json.load(f)

    # Pin all "control*" variants first (control, control_reversed, etc.),
    # then everything else in JSON insertion order.
    names = list(r["conditions"].keys())
    controls = [n for n in names if n.startswith("control")]
    others = [n for n in names if not n.startswith("control")]
    names = controls + others

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

    ax.set_ylabel("P(5 | answer in {5, 7})")
    ax.set_ylim(0, 1)
    ax.set_title(r["model"])

    for xi, pi in zip(x, p_logit):
        ax.text(xi, pi + 0.02, f"{pi:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = Path(args.results_json).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
