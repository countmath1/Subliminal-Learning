"""Operator-randomization experiment.

For each list length L: freeze the L animal *pairs* (items fixed, original
operators discarded), then run N iterations. Each iteration draws a fresh
operator vector — an independent fair coin (`<` or `>`) per line — builds
the prompt, and measures P(5 | {5,7}) via the one-forward-pass logit.

Within an L the items are frozen, so the *variance* across iterations is the
pure operator-direction effect. Across L (or across shuffle seeds) the items
change, so that variation reflects item content.

Supports running many shuffle seeds in one process (--shuffle-seeds) so the
model load + vocab scan are amortized; each seed's output goes to a
seed_<N>/ subdirectory of --out.

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
    # Render the chat template to a string, then tokenize to a plain
    # list[int]. apply_chat_template(tokenize=True) returns a
    # tokenizers.Encoding in transformers 5.x, which torch.tensor can't take.
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
        out = model(x)
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


def measure_iters(model, tok, prompt_pairs, n_iters, L, op_rng, ids_5, ids_7,
                  device, batch_tokens, preamble, question, enable_thinking):
    """Run n_iters random-operator measurements for a fixed pair set."""
    ops_matrix = op_rng.integers(0, 2, size=(n_iters, L))
    id_lists = [encode(tok, build_prompt(preamble, prompt_pairs, ops_matrix[i],
                                         question), enable_thinking)
                for i in range(n_iters)]
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
    return ops_matrix, p5_arr, lo_arr


def run_one_seed(model, tok, ids_5, ids_7, pairs_all, cfg, shuffle_seed,
                 out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    enable_thinking = cfg.get("enable_thinking", False)
    n_iters = cfg["n_iters"]
    batch_tokens = cfg.get("batch_tokens", 24000)
    preamble = cfg["preamble"]
    question = cfg["question"]

    shuf = np.random.default_rng(shuffle_seed).permutation(len(pairs_all))
    pairs_shuffled = [pairs_all[i] for i in shuf]
    op_rng = np.random.default_rng(cfg["op_seed"])

    results = {
        "model": cfg["model"],
        "enable_thinking": enable_thinking,
        "n_iters": n_iters,
        "shuffle_seed": shuffle_seed,
        "op_seed": cfg["op_seed"],
        "list_file": cfg["list_file"],
        "results": {},
    }

    # Bare control (question only, no preamble/list).
    ctrl_prompt = question.rstrip() + "\n"
    cp5, clo = score_batch(model, [encode(tok, ctrl_prompt, enable_thinking)],
                           ids_5, ids_7, device)
    results["control"] = {"p5": cp5[0], "log_odds": clo[0], "prompt": ctrl_prompt}

    sidecar = open(out_dir / "operator_rand_iters.jsonl", "w")
    seen_L = set()
    for L_req in cfg["L_values"]:
        L = min(L_req, len(pairs_shuffled))
        if L in seen_L:
            continue
        seen_L.add(L)
        pairs = pairs_shuffled[:L]
        ops_matrix, p5_arr, lo_arr = measure_iters(
            model, tok, pairs, n_iters, L, op_rng, ids_5, ids_7, device,
            batch_tokens, preamble, question, enable_thinking)
        for i in range(n_iters):
            ops_str = "".join(">" if o else "<" for o in ops_matrix[i])
            sidecar.write(json.dumps({
                "L": L, "iter": i, "ops": ops_str,
                "p5": p5_arr[i], "log_odds": lo_arr[i],
            }) + "\n")
        arr = np.array(p5_arr)
        results["results"][str(L)] = {
            "p5": p5_arr, "log_odds": lo_arr,
            "mean_p5": float(arr.mean()), "std_p5": float(arr.std()),
            "min_p5": float(arr.min()), "max_p5": float(arr.max()),
            "quantiles": {str(q): float(np.quantile(arr, q))
                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        }
        print(f"  seed={shuffle_seed} L={L:>4}  mean_p5={arr.mean():.4f}  "
              f"std={arr.std():.4f}", flush=True)
    sidecar.close()
    with open(out_dir / "operator_rand_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def parse_seeds(s):
    """'0-9' -> range, '0,1,2' -> list, '0-3,7' -> mixed."""
    seeds = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-seeds", default=None,
                    help="e.g. '0-9' or '0,8,16'. Each seed -> seed_<N>/ subdir. "
                         "If omitted, uses cfg['shuffle_seed'] and writes to --out.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg['model']} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map=device)
    model.eval()

    ids_5 = candidate_first_tokens(tok, 5)
    ids_7 = candidate_first_tokens(tok, 7)
    assert ids_5 and ids_7
    pairs_all = parse_pairs(cfg["list_file"])
    print(f"{len(pairs_all)} pairs in pool", flush=True)

    if args.shuffle_seeds:
        seeds = parse_seeds(args.shuffle_seeds)
        print(f"running {len(seeds)} seeds: {seeds}", flush=True)
        for sd in seeds:
            print(f"=== seed {sd} ===", flush=True)
            run_one_seed(model, tok, ids_5, ids_7, pairs_all, cfg, sd,
                         Path(args.out) / f"seed_{sd}", device)
    else:
        run_one_seed(model, tok, ids_5, ids_7, pairs_all, cfg,
                     cfg["shuffle_seed"], Path(args.out), device)

    print("done", flush=True)


if __name__ == "__main__":
    main()
