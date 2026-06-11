"""Thinking-mode operator-randomization on a single fixed list.

Like exp_operator_rand.py (fixed items, randomize only the `<`/`>` operators
across iterations), but with Qwen3 thinking ENABLED. In thinking mode the
answer is conditioned on a sampled reasoning trace, so there is no
one-forward-pass logit: each iteration generates a full trace and we parse
the committed 5-vs-7 answer from it. Every trace is saved for auditing
(e.g. whether the reasoning references the preference list).

One fixed item set (the whole pool, or the first L pairs); N iterations,
each a fresh random operator vector + one sampled generation.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_pairs(path):
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
    lines = [f"{l} {'>' if o else '<'} {r}" for (l, r), o in zip(pairs, ops)]
    return preamble.rstrip() + "\n\n" + "\n".join(lines) + "\n\n" + question.rstrip() + "\n"


def encode(tok, prompt, enable_thinking):
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking,
        tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def parse_answer(text):
    """Return (answer, has_think_close); answer in {'5','7','other'}.

    1) strip <think>...</think>; 2) honor 'Final answer: N'; 3) else last
    standalone 5 or 7 in the answer portion; 4) else 'other'.
    """
    has_close = "</think>" in text
    portion = re.sub(r".*</think>", "", text, flags=re.DOTALL) if has_close else text
    m = re.search(r"final answer[:\s]*\**\s*(\d+)", portion, flags=re.IGNORECASE)
    if m:
        d = m.group(1)
        return (d if d in ("5", "7") else "other"), has_close
    standalone = re.findall(r"(?<!\d)([57])(?!\d)", portion)
    if standalone:
        return standalone[-1], has_close
    return "other", has_close


def left_pad(id_lists, pad_id):
    maxlen = max(len(x) for x in id_lists)
    ids = [[pad_id] * (maxlen - len(x)) + x for x in id_lists]
    attn = [[0] * (maxlen - len(x)) + [1] * len(x) for x in id_lists]
    return ids, attn


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
        cfg["model"], dtype=torch.bfloat16, device_map=device)
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id

    enable_thinking = cfg.get("enable_thinking", True)
    n_iters = cfg["n_iters"]
    max_new_tokens = cfg["max_new_tokens"]
    chunk_size = cfg.get("sample_chunk_size", 16)
    gen_kwargs = cfg.get("gen", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    preamble = cfg["preamble"]
    question = cfg["question"]

    pairs_all = parse_pairs(cfg["list_file"])
    L = min(cfg.get("L", len(pairs_all)), len(pairs_all))
    pairs = pairs_all[:L]
    print(f"{len(pairs_all)} pairs in pool; using L={L} (fixed items)", flush=True)

    op_rng = np.random.default_rng(cfg["op_seed"])
    ops_matrix = op_rng.integers(0, 2, size=(n_iters, L))

    counts = {"5": 0, "7": 0, "other": 0}
    n_truncated = 0
    n_no_close = 0
    gen_lens = []
    trace_path = out_dir / "exp_reason_operator_traces.jsonl"
    tf = open(trace_path, "w")

    i = 0
    while i < n_iters:
        cn = min(chunk_size, n_iters - i)
        id_lists = [encode(tok, build_prompt(preamble, pairs, ops_matrix[i + j], question),
                           enable_thinking) for j in range(cn)]
        ids, attn = left_pad(id_lists, pad_id)
        input_ids = torch.tensor(ids, device=device)
        attention = torch.tensor(attn, device=device)
        with torch.inference_mode():
            out = model.generate(
                input_ids, attention_mask=attention, do_sample=True,
                max_new_tokens=max_new_tokens, pad_token_id=pad_id, **gen_kwargs)
        new = out[:, input_ids.shape[1]:]
        for j, row in enumerate(new):
            eos_pos = (row == eos_id).nonzero()
            truncated = len(eos_pos) == 0
            gen_len = eos_pos[0].item() if len(eos_pos) else row.shape[0]
            text = tok.decode(row, skip_special_tokens=True)
            ans, has_close = parse_answer(text)
            counts[ans] += 1
            n_truncated += truncated
            n_no_close += (not has_close)
            gen_lens.append(gen_len)
            ops_str = "".join(">" if o else "<" for o in ops_matrix[i + j])
            tf.write(json.dumps({
                "iter": i + j, "answer": ans, "has_think_close": has_close,
                "truncated": truncated, "gen_len": gen_len, "ops": ops_str,
                "text": text,
            }) + "\n")
        i += cn
        n57 = counts["5"] + counts["7"]
        p5 = counts["5"] / n57 if n57 else float("nan")
        print(f"  {i}/{n_iters}  counts={counts}  P(5|5,7)={p5:.3f}  "
              f"trunc={n_truncated}  mean_len={np.mean(gen_lens):.0f}", flush=True)
    tf.close()

    n57 = counts["5"] + counts["7"]
    results = {
        "model": cfg["model"],
        "enable_thinking": enable_thinking,
        "n_iters": n_iters,
        "L": L,
        "op_seed": cfg["op_seed"],
        "max_new_tokens": max_new_tokens,
        "gen": gen_kwargs,
        "counts": counts,
        "p_5_given_5_or_7": (counts["5"] / n57) if n57 else None,
        "n_truncated": n_truncated,
        "n_no_think_close": n_no_close,
        "mean_gen_len": float(np.mean(gen_lens)),
        "max_gen_len": int(np.max(gen_lens)),
    }
    with open(out_dir / "exp_reason_operator_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_dir / 'exp_reason_operator_results.json'} and {trace_path}",
          flush=True)


if __name__ == "__main__":
    main()
