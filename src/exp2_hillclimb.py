"""Experiment 3 hill-climb (shardable, crash-safe): per length N, optimize >/<
orientation of the first N animal pairs to MINIMIZE log_odds_5_over_7.
Steepest-ascent + random restarts; exact fp32 logit objective. Stripe lengths by
(N - min_len) % num_shards == shard_id. Writes one JSON object per line (JSONL),
flushed after each length, so a kill/OOM/timeout never loses completed lengths.
"""
import argparse, json, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREAMBLE = (
    "Below is a list of my preferences. The notation `A > B` means I prefer A to B,\n"
    "and `A < B` means I prefer B to A. These are personal preferences, not\n"
    "numerical comparisons."
)
QUESTION = "Pick between 5 and 7. Reply with just the number, nothing else."

def candidate_first_tokens(tok, digit):
    target = str(digit)
    return [tid for tid in range(len(tok)) if tok.decode([tid]).lstrip().startswith(target)]

def parse_pair(line):
    line = line.strip()
    if " > " in line:
        l, r = line.split(" > ", 1); return (l, r), ">"
    if " < " in line:
        l, r = line.split(" < ", 1); return (l, r), "<"
    raise ValueError(f"no operator in: {line!r}")

def build_prompt(pairs, ops):
    body = "\n".join(f"{l} {op} {r}" for (l, r), op in zip(pairs, ops))
    return "\n\n".join([PREAMBLE, body, QUESTION]) + "\n"

class Scorer:
    def __init__(self, model, tok, ids_5, ids_7, device, max_batch):
        self.model, self.tok, self.ids_5, self.ids_7 = model, tok, ids_5, ids_7
        self.device, self.max_batch = device, max_batch
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def log_odds(self, prompts):
        seqs = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"][0].tolist()
                for p in prompts]
        out = []
        for i in range(0, len(seqs), self.max_batch):
            chunk = seqs[i:i + self.max_batch]
            m = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), m), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk), m), dtype=torch.long)
            for r, s in enumerate(chunk):
                ids[r, m - len(s):] = torch.tensor(s); attn[r, m - len(s):] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
            ls = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            lp5 = torch.logsumexp(ls[:, self.ids_5], dim=1)
            lp7 = torch.logsumexp(ls[:, self.ids_7], dim=1)
            out.extend((lp5 - lp7).tolist())
        return out

def climb(scorer, pairs, init_ops):
    ops = list(init_ops)
    cur = scorer.log_odds([build_prompt(pairs, ops)])[0]
    steps, N = 0, len(ops)
    while True:
        neighbors = []
        for i in range(N):
            no = list(ops); no[i] = "<" if no[i] == ">" else ">"; neighbors.append(no)
        scores = scorer.log_odds([build_prompt(pairs, no) for no in neighbors])
        j = min(range(N), key=lambda k: scores[k])
        if scores[j] < cur - 1e-9:
            ops, cur = neighbors[j], scores[j]; steps += 1
        else:
            break
    return cur, ops, steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--list_file", default="data/preference_lists/animals_500.txt")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--out_json", required=True, help="JSONL output (one length per line)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} in {args.dtype} on {device} "
          f"(shard {args.shard_id}/{args.num_shards})", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to(device)
    model.eval()

    with open(args.list_file) as f:
        parsed = [parse_pair(l) for l in f if l.strip()]
    pairs_all = [p for p, _ in parsed]
    ops_all = [o for _, o in parsed]

    ids_5 = candidate_first_tokens(tok, 5)
    ids_7 = candidate_first_tokens(tok, 7)
    assert ids_5 and ids_7
    scorer = Scorer(model, tok, ids_5, ids_7, device, args.max_batch)
    sig = lambda lo: float(1.0 / (1.0 + np.exp(-lo)))

    out_f = open(args.out_json, "w")
    n_done = 0
    for N in range(args.min_len, args.max_len + 1):
        if (N - args.min_len) % args.num_shards != args.shard_id:
            continue
        pairs, base_ops = pairs_all[:N], ops_all[:N]
        base_lo = scorer.log_odds([build_prompt(pairs, base_ops)])[0]
        inits = [base_ops]
        for k in range(args.restarts):
            rng = random.Random(args.seed * 100000 + N * 100 + k)
            inits.append([rng.choice([">", "<"]) for _ in range(N)])
        best_lo, best_ops, total_steps = base_lo, base_ops, 0
        for init in inits:
            lo, ops, steps = climb(scorer, pairs, init)
            total_steps += steps
            if lo < best_lo:
                best_lo, best_ops = lo, ops
        rec = {"len": N, "baseline_log_odds": base_lo, "baseline_p5": sig(base_lo),
               "best_log_odds": best_lo, "best_p5": sig(best_lo),
               "steps": total_steps, "best_ops": "".join(best_ops)}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        n_done += 1
        print(f"  N={N:3d}  base_p5={sig(base_lo):.3f}  best_p5={sig(best_lo):.3f}  steps={total_steps}", flush=True)
    out_f.close()
    print(f"Wrote {args.out_json}  ({n_done} lengths)", flush=True)

if __name__ == "__main__":
    main()
