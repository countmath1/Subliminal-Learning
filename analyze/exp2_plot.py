"""Experiment 2: P(5 | 5 or 7) vs animal-list length.

Reads exp1_results.json (from src/exp1_infer.py on configs/exp2.yaml), parses
the length L from each `len_<L>` condition name, and plots the logit metric
p_5_given_5_or_7 against L. Single series, x in [0, 500].
"""
import argparse, json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True, help="path to exp1_results.json")
ap.add_argument("--out", default="analyze/exp2_p5.png")
args = ap.parse_args()

conds = json.load(open(args.results))["conditions"]
pts = []
for name, data in conds.items():
    m = re.fullmatch(r"len_(\d+)", name)
    if m:
        pts.append((int(m.group(1)), data["logit"]["p_5_given_5_or_7"]))
pts.sort()
xs, ys = [p[0] for p in pts], [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(xs, ys, marker="o", color="tab:blue")
ax.set_xlim(0, 500)
ax.set_xlabel("animal-list length (number of preferences)")
ax.set_ylabel(r"$P(5 \mid 5\ \mathrm{or}\ 7)$")
ax.set_title("Qwen2.5-3B-Instruct: preference for 5 vs list length")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"Wrote {args.out}  ({len(xs)} points: {xs})")
