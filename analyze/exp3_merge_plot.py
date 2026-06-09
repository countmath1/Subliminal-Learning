"""Merge per-shard JSONL hill-climb outputs (and/or scrape logs) into one
result + overlay plot: default vs hill-climbed P(5|5or7) across length."""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True, help="dir with shard_*.jsonl")
ap.add_argument("--out_json", required=True); ap.add_argument("--out_png", required=True)
a = ap.parse_args()

d = {}
for f in glob.glob(os.path.join(a.in_dir, "shard_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if line:
            r = json.loads(line); d[r["len"]] = r
pts = [d[k] for k in sorted(d)]
json.dump({"points": pts}, open(a.out_json, "w"), indent=2)

xs = [p["len"] for p in pts]; base = [p["baseline_p5"] for p in pts]; best = [p["best_p5"] for p in pts]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(xs, base, color="0.6", lw=1, label="default orientation")
ax.plot(xs, best, color="tab:red", lw=1.5, label="hill-climbed")
ax.axhline(0.5, color="0.8", ls="--", lw=0.8)
ax.set_xlim(min(xs) if xs else 1, max(xs) if xs else 100)
ax.set_xlabel("animal-list length (number of preferences)"); ax.set_ylabel(r"$P(5 \mid 5\ \mathrm{or}\ 7)$")
ax.set_title("Orientation hill-climb vs default (Qwen2.5-3B, fp32)"); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(a.out_png, dpi=150)

print(f"lengths merged: {len(pts)} (expect 100); total steps: {sum(p['steps'] for p in pts)}")
if pts:
    bi = min(range(len(pts)), key=lambda i: best[i])
    print(f"best hill-climbed: P5={best[bi]:.3f} at L={xs[bi]} (default there {base[bi]:.3f})")
