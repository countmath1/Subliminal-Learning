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
                merged["results"][L] = {"counts": {"5": 0, "7": 0, "other": 0},
                                        "n_forced": 0, "n_iters": 0}
                gen_lens[L] = []
            m = merged["results"][L]
            for k in ("5", "7", "other"):
                m["counts"][k] += res["counts"][k]
            m["n_forced"] += res.get("n_forced", 0)
            m["n_iters"] += res.get("n_iters", 0)
            gen_lens[L].append(res.get("mean_gen_len", 0))

    for L, m in merged["results"].items():
        n57 = m["counts"]["5"] + m["counts"]["7"]
        m["p_5_given_5_or_7"] = (m["counts"]["5"] / n57) if n57 else None
        m["n_5_or_7"] = n57
        m["mean_gen_len"] = sum(gen_lens[L]) / len(gen_lens[L]) if gen_lens[L] else 0

    out = Path(args.parent_dir) / "exp_reason_operator_results.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Wrote {out}  ({merged['n_tasks']} tasks merged)")
    for L in sorted(merged["results"], key=int):
        m = merged["results"][L]
        print(f"  L={L:>4}  n={m['n_iters']}  P(5|5,7)={m['p_5_given_5_or_7']:.3f}  "
              f"forced={m['n_forced']}")


if __name__ == "__main__":
    main()
