"""Plot the per-condition distribution of first-token digits emitted.

For each condition, one subplot showing how often the model's first
sampled token was each digit 0..9 vs a non-digit. The 5 and 7 bars are
highlighted to indicate the intended answer space.

Requires JSON written by `exp1_infer.py` after the per-digit refactor —
i.e., each condition's `sampling` must have a `digit_counts` field.

Usage:  py analyze/exp1_digit_dist.py runs/<file>.json
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
    args = ap.parse_args()

    with open(args.results_json) as f:
        r = json.load(f)

    # Pin "control*" first, then the rest in JSON order.
    names = list(r["conditions"].keys())
    controls = [n for n in names if n.startswith("control")]
    others = [n for n in names if not n.startswith("control")]
    names = controls + others

    cats = [str(d) for d in range(10)] + ["non-digit"]

    n_conds = len(names)
    fig, axes = plt.subplots(n_conds, 1, figsize=(9, 2.2 * n_conds), sharex=True)
    if n_conds == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        sampling = r["conditions"][name]["sampling"]
        dc = sampling.get("digit_counts")
        if dc is None:
            ax.text(0.5, 0.5, f"no digit_counts — re-run with newer exp1_infer.py",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, loc="left")
            continue
        n = sampling["n"]
        counts = [dc.get(str(d), 0) for d in range(10)] + [dc.get("non_digit", 0)]
        freqs = [c / n for c in counts]

        colors = ["lightgrey"] * 11
        colors[5] = "C2"   # highlight 5
        colors[7] = "C3"   # highlight 7
        bars = ax.bar(cats, freqs, color=colors)

        # Numeric labels above any nonzero bar
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        str(count), ha="center", va="bottom", fontsize=8)

        ax.set_ylim(0, max(max(freqs) * 1.2, 0.05))
        ax.set_ylabel(r"$P(\mathrm{first\ token})$")
        ax.set_title(rf"$\mathrm{{{name.replace('_', r'\_')}}}$  ($n={n}$)", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel(r"first-token category")
    fig.suptitle(r["model"], y=1.0)
    plt.tight_layout()

    src = Path(args.results_json)
    out = src.parent / (src.stem + "_digit_dist.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
