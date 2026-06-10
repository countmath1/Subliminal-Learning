"""Operator-randomization experiment.

For each list length L: freeze the L animal *pairs* (items fixed, original
operators discarded), then run N iterations. Each iteration draws a fresh
operator vector — an independent fair coin (`<` or `>`) per line — builds
the prompt, and measures P(5 | {5,7}) via the one-forward-pass logit.

Because the items are frozen within an L and only the directions vary, the
*variance* across iterations isolates the pure operator-direction effect,
while the *mean* is the combined "animal words present" + average-direction
effect. The resulting per-L distributions are the random-operator baseline
that later coordinate descent (exp2) must beat to claim it found structure.

Measurement uses Qwen3 with thinking disabled (enable_thinking=False) so the
first generated token is the answer and the logit shortcut is valid.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def candidate_first_tokens(tok, digit):
    """Vocab ids whose decoded string starts with `digit` after lstrip."""
    target = str(digit)
    return [tid for tid in range(len(tok))
            if tok.decode([tid]).lstrip().startswith(target)]


def parse_pairs(path):
    """Parse 'left OP right' lines into (left, right), discarding operators."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s*[<>]\s*", line)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def build_prompt(preamble, pairs, ops, question):
    """ops: array of {0,1}; 1 -> '>', 0 -> '<'."""
    lines = [f"{l} {'>' if o else '<'} {r}" for (l, r), o in zip(pairs, ops)]
    body = "\n".join(lines)
    return preamble.rstrip() + "\n\n" + body + "\n\n" + question.rstrip() + "\n"


def encode(tok, prompt, enable_thinking):
    # Two-step: render the chat template to a string, then tokenize to a
    # plain list[int]. apply_chat_template(tokenize=True) returns a
    # tokenizers.Encoding in transformers 5.x, which torch.tensor can't
    # consume; add_special_tokens=False avoids double-adding specials the
    # template already includes.
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking,
        tokenize=False,
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def last_logits(model, x):
    """Last-position logits only, to avoid materializing [B, seq, vocab]."""
    try:
        out = model(x, logits_to_keep=1)
    except TypeError:
        out = model(x)  # older transformers without logits_to_keep
    return out.logits[:, -1, :].float()


def score_batch(model, id_lists, ids_5, ids_7, device):
    """All id_lists must share the same length (caller buckets by length)."""
    x = torch.tensor(id_lists, device=device)
    with torch.inference_mode():
        logits = last_logits(model, x)
    log_sm = torch.log_softmax(logits, dim=-1)
    lp5 = torch.logsumexp(log_sm[:, ids_5], dim=-1)
    lp7 = torch.logsumexp(log_sm[:, ids_7], dim=-1)
    p5 = (lp5.exp() / (lp5.exp() + lp7.exp())).tolist()
    log_odds = (lp5 - lp7).tolist()
    return p5, log_odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg['model']} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    ids_5 = candidate_first_tokens(tok, 5)
    ids_7 = candidate_first_tokens(tok, 7)
    assert ids_5 and ids_7

    enable_thinking = cfg.get("enable_thinking", False)
    n_iters = cfg["n_iters"]
    batch_tokens = cfg.get("batch_tokens", 24000)
    preamble = cfg["preamble"]
    question = cfg["question"]

    pairs_all = parse_pairs(cfg["list_file"])
    shuf = np.random.default_rng(cfg["shuffle_seed"]).permutation(len(pairs_all))
    pairs_shuffled = [pairs_all[i] for i in shuf]
    print(f"{len(pairs_all)} pairs in pool; shuffled with seed {cfg['shuffle_seed']}",
          flush=True)

    op_rng = np.random.default_rng(cfg["op_seed"])

    results = {
        "model": cfg["model"],
        "enable_thinking": enable_thinking,
        "n_iters": n_iters,
        "shuffle_seed": cfg["shuffle_seed"],
        "op_seed": cfg["op_seed"],
        "list_file": cfg["list_file"],
        "results": {},
    }

    sidecar_path = out_dir / "operator_rand_iters.jsonl"
    sidecar = open(sidecar_path, "w")

    seen_L = set()
    for L_req in cfg["L_values"]:
        L = min(L_req, len(pairs_shuffled))
        if L in seen_L:
            print(f"skipping duplicate L={L} (requested {L_req})", flush=True)
            continue
        seen_L.add(L)
        pairs = pairs_shuffled[:L]

        ops_matrix = op_rng.integers(0, 2, size=(n_iters, L))

        # Encode every iteration's prompt.
        id_lists = [encode(tok, build_prompt(preamble, pairs, ops_matrix[i], question),
                           enable_thinking)
                    for i in range(n_iters)]

        # Bucket by exact token length so each sub-batch is equal-length and
        # needs no padding (keeps RoPE position_ids correct).
        buckets = defaultdict(list)
        for idx, ids in enumerate(id_lists):
            buckets[len(ids)].append(idx)

        p5_arr = [None] * n_iters
        lo_arr = [None] * n_iters
        for length, idxs in buckets.items():
            bs = max(1, batch_tokens // length)
            for c in range(0, len(idxs), bs):
                chunk = idxs[c:c + bs]
                p5s, los = score_batch(model, [id_lists[j] for j in chunk],
                                       ids_5, ids_7, device)
                for j, p5v, lov in zip(chunk, p5s, los):
                    p5_arr[j] = p5v
                    lo_arr[j] = lov

        for i in range(n_iters):
            ops_str = "".join(">" if o else "<" for o in ops_matrix[i])
            sidecar.write(json.dumps({
                "L": L, "iter": i, "ops": ops_str,
                "p5": p5_arr[i], "log_odds": lo_arr[i],
            }) + "\n")

        arr = np.array(p5_arr)
        results["results"][str(L)] = {
            "p5": p5_arr,
            "log_odds": lo_arr,
            "mean_p5": float(arr.mean()),
            "std_p5": float(arr.std()),
            "min_p5": float(arr.min()),
            "max_p5": float(arr.max()),
            "quantiles": {str(q): float(np.quantile(arr, q))
                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        }
        print(f"L={L:>4}  mean_p5={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}", flush=True)

    sidecar.close()
    out_path = out_dir / "operator_rand_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path} and {sidecar_path}", flush=True)


if __name__ == "__main__":
    main()
