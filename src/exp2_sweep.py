"""Experiment 2 (dense): P(5 | 5 or 7) for EVERY list length 0..N.

Logit-only (one forward pass per length, no sampling) and fp32 (avoids the
bf16 staircase that made lengths 20/34/50 read bit-identical). Writes a JSON
of per-length results AND a PNG. Prompt assembly matches configs/exp2.yaml:
  L=0  -> question only (control);  L>=1 -> preamble + first L prefs + question
"""
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREAMBLE = (
    "Below is a list of my preferences. The notation `A > B` means I prefer A to B,\n"
    "and `A < B` means I prefer B to A. These are personal preferences, not\n"
    "numerical comparisons."
)
QUESTION = "Pick between 5 and 7. Reply with just the number, nothing else."

def candidate_first_tokens(tok, digit):
    target = str(digit)
    return [tid for tid in range(len(tok)) if tok.decode([tid]).lstrip().startswith(target)]

def build_prompt(list_lines, L):
    if L == 0:
        return QUESTION + "\n"
    body = "".join(list_lines[:L]).rstrip()
    return "\n\n".join([PREAMBLE, body, QUESTION]) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--list_file", default="data/preference_lists/animals_500.txt")
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} in {args.dtype} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    with open(args.list_file) as f:
        list_lines = f.readlines()
    max_len = min(args.max_len, len(list_lines))

    ids_5 = candidate_first_tokens(tok, 5)
    ids_7 = candidate_first_tokens(tok, 7)
    assert ids_5 and ids_7, "no candidate tokens for 5 or 7"

    points = []
    for L in range(0, max_len + 1):
        prompt = build_prompt(list_lines, L)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids)
        log_sm = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        logp_5 = torch.logsumexp(log_sm[ids_5], dim=0).item()
        logp_7 = torch.logsumexp(log_sm[ids_7], dim=0).item()
        p5 = float(np.exp(logp_5) / (np.exp(logp_5) + np.exp(logp_7)))
        points.append({"len": L, "logp_5": logp_5, "logp_7": logp_7,
                       "p_5_given_5_or_7": p5, "log_odds_5_over_7": logp_5 - logp_7})
        if L % 25 == 0:
            print(f"  L={L:3d}  p5={p5:.4f}", flush=True)

    result = {"model": args.model, "dtype": args.dtype, "list_file": args.list_file,
              "preamble": PREAMBLE, "question": QUESTION, "points": points}
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out_json}", flush=True)

    xs = [p["len"] for p in points]
    ys = [p["p_5_given_5_or_7"] for p in points]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, color="tab:blue", linewidth=1)
    ax.set_xlim(0, args.max_len)
    ax.set_xlabel("animal-list length (number of preferences)")
    ax.set_ylabel(r"$P(5 \mid 5\ \mathrm{or}\ 7)$")
    ax.set_title(f"{args.model}: preference for 5 vs list length ({args.dtype})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote {args.out_png}", flush=True)

if __name__ == "__main__":
    main()
