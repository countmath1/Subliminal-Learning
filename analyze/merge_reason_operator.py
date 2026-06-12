"""Merge per-task results from the thinking-mode operator array.

Each array task wrote task_<id>/exp_reason_operator_results.json with its
share of the iterations. This sums the 5/7/other counts per L across tasks
into one combined results JSON (same schema as a single run) that the
existing exp_reason_operator_plot.py can read.

Usage:
  python analyze/merge_reason_operator.py <parent_dir>
"""
import argparse
import glob
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent_dir")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.parent_dir, "task_*",
                                          "exp_reason_operator_results.json")))
    if not paths:
        raise SystemExit(f"no task_*/exp_reason_operator_results.json under {args.parent_dir}")

    merged = None
    gen_lens = {}
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        if merged is None:
            merged = {k: v for k, v in r.items() if k != "results"}
            merged["results"] = {}
            merged["n_tasks"] = 0
        merged["n_tasks"] += 1
        for L, res in r["results"].items():
            if L not in merged["results"]:
                merged["results"][L] = {
                    "counts": {"5": 0, "7": 0, "other": 0},
                    "counts_natural": {"5": 0, "7": 0, "other": 0},
                    "counts_forced": {"5": 0, "7": 0, "other": 0},
                    "n_forced": 0, "n_iters": 0}
                gen_lens[L] = []
            m = merged["results"][L]
            for grp in ("counts", "counts_natural", "counts_forced"):
                for k in ("5", "7", "other"):
                    m[grp][k] += res.get(grp, {}).get(k, 0)
            m["n_forced"] += res.get("n_forced", 0)
            m["n_iters"] += res.get("n_iters", 0)
            gen_lens[L].append(res.get("mean_gen_len", 0))

    def p5(c):
        n = c["5"] + c["7"]
        return (c["5"] / n) if n else None

    for L, m in merged["results"].items():
        m["p_5_given_5_or_7"] = p5(m["counts"])
        m["p_5_natural"] = p5(m["counts_natural"])
        m["p_5_forced"] = p5(m["counts_forced"])
        m["n_5_or_7"] = m["counts"]["5"] + m["counts"]["7"]
        m["mean_gen_len"] = sum(gen_lens[L]) / len(gen_lens[L]) if gen_lens[L] else 0

    out = Path(args.parent_dir) / "exp_reason_operator_results.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out}  ({merged['n_tasks']} tasks merged)")
    for L in sorted(merged["results"], key=int):
        m = merged["results"][L]
        nat = m["counts_natural"]["5"] + m["counts_natural"]["7"]
        pf = m["p_5_forced"]
        pn = m["p_5_natural"]
        print(f"  L={L:>4}  n={m['n_iters']}  all={m['p_5_given_5_or_7']:.3f}  "
              f"natural={pn:.3f} (n={nat})  "
              f"forced={pf if pf is None else round(pf,3)} (n_forced={m['n_forced']})")


if __name__ == "__main__":
    main()
