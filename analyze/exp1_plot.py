"""Plot exp1 results: P(5 | answer in {5,7}) for control vs treatment, logit
and sampling estimates side by side, with a 50/50 reference line.

Usage:  python analyze/exp1_plot.py runs/<jobid>/exp1_results.json
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

    names = ["control", "treatment"]
    p_logit = [r["conditions"][n]["logit"]["p_5_given_5_or_7"] for n in names]

    p_samp, se_samp = [], []
    for n in names:
        c5 = r["conditions"][n]["sampling"]["counts"]["5"]
        c7 = r["conditions"][n]["sampling"]["counts"]["7"]
        denom = max(c5 + c7, 1)
        phat = c5 / denom
        p_samp.append(phat)
        se_samp.append(np.sqrt(phat * (1 - phat) / denom))

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, p_logit, w, label="logit-based P(5 | {5,7})")
    ax.bar(x + w/2, p_samp, w, yerr=se_samp, capsize=4,
           label="sampled freq (T=1)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("P(5 | answer in {5,7})")
    ax.set_ylim(0, 1)
    ax.set_title(f"{r['model']} — n_samples={r['n_samples']}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out = Path(args.results_json).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
