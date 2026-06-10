"""Plot the operator-randomization results.

Two panels:
  (top)    Distribution of P(5 | {5,7}) over 1000 random operator vectors,
           one violin/box per list length L. Shows how the spread (pure
           operator-direction effect) and center evolve with L.
  (bottom) Mean P(5) +/- 1 std vs L on a log-x axis, with reference lines
           for the bare control and the hand-curated 34-item treatment.

Usage:  py analyze/operator_rand_plot.py runs/<file>.json
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

# Reference values from the exp1 logit runs (Qwen2.5-3B; shown for context).
CONTROL_P5 = 0.9859
TREATMENT34_P5 = 0.6792


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--refs", action="store_true",
                    help="overlay exp1 control / treatment-34 reference lines")
    args = ap.parse_args()

    with open(args.results_json) as f:
        r = json.load(f)

    Ls = sorted(int(k) for k in r["results"].keys())
    data = [np.array(r["results"][str(L)]["p5"]) for L in Ls]
    means = np.array([r["results"][str(L)]["mean_p5"] for L in Ls])
    stds = np.array([r["results"][str(L)]["std_p5"] for L in Ls])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7))

    # --- top: distributions ---
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
    ax1.set_ylabel(r"$P(5 \mid \mathrm{answer} \in \{5,7\})$")
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax1.set_title("Distribution over 1000 random operator vectors")

    # --- bottom: mean +/- std vs L (log x) ---
    ax2.errorbar(Ls, means, yerr=stds, marker="o", capsize=3,
                 color="C0", label=r"mean $\pm$ 1 std")
    if args.refs:
        ax2.axhline(CONTROL_P5, color="grey", linestyle=":", linewidth=1,
                    label="exp1 control")
        ax2.axhline(TREATMENT34_P5, color="C3", linestyle=":", linewidth=1,
                    label="exp1 treatment (34)")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"list length $L$ (log scale)")
    ax2.set_ylabel(r"$P(5 \mid \{5,7\})$")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="best")
    ax2.set_title(r"Mean priming effect vs $L$")

    fig.suptitle(r["model"])
    plt.tight_layout()
    out = Path(args.results_json).with_suffix(".png")
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
