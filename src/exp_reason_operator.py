"""Thinking-mode operator-randomization across list lengths.

For each L in L_values: freeze the first L animal pairs (items fixed), then
run n_iters iterations. Each iteration draws a fresh random `<`/`>` vector,
generates one Qwen3 reasoning trace (thinking ON), and parses the committed
5-vs-7 answer. Aggregating the binary answers gives ONE proportion
P(5 | {5,7}) per L (a point, with a binomial CI computed at plot time) --
the thinking-mode analog of the logit L-sweep, minus the per-vector violin
(which thinking mode can't produce without many samples per vector).

Every trace is saved (capped per L) for auditing the reasoning.
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


def measure_L(model, tok, pairs, L, n_iters, op_rng, gen_kwargs, max_new_tokens,
              chunk_size, preamble, question, enable_thinking, device,
              pad_id, eos_id, tf, save_cap):
    counts = {"5": 0, "7": 0, "other": 0}
    n_truncated = 0
    n_no_close = 0
    gen_lens = []
    saved = 0
    ops_matrix = op_rng.integers(0, 2, size=(n_iters, L))
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
            if saved < save_cap:
                tf.write(json.dumps({
                    "L": L, "iter": i + j, "answer": ans,
                    "has_think_close": has_close, "truncated": truncated,
                    "gen_len": gen_len,
                    "ops": "".join(">" if o else "<" for o in ops_matrix[i + j]),
                    "text": text,
                }) + "\n")
                saved += 1
        i += cn
        n57 = counts["5"] + counts["7"]
        p5 = counts["5"] / n57 if n57 else float("nan")
        print(f"  L={L:>4} {i}/{n_iters}  counts={counts}  P(5|5,7)={p5:.3f}  "
              f"trunc={n_truncated}  mean_len={np.mean(gen_lens):.0f}", flush=True)
    n57 = counts["5"] + counts["7"]
    return {
        "counts": counts,
        "n_iters": n_iters,
        "p_5_given_5_or_7": (counts["5"] / n57) if n57 else None,
        "n_5_or_7": n57,
        "n_truncated": n_truncated,
        "n_no_think_close": n_no_close,
        "mean_gen_len": float(np.mean(gen_lens)),
        "max_gen_len": int(np.max(gen_lens)),
    }


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
    save_cap = cfg.get("trace_save_cap", 100)

    pairs_all = parse_pairs(cfg["list_file"])
    print(f"{len(pairs_all)} pairs in pool", flush=True)
    op_rng = np.random.default_rng(cfg["op_seed"])

    results = {
        "model": cfg["model"], "enable_thinking": enable_thinking,
        "n_iters": n_iters, "op_seed": cfg["op_seed"],
        "max_new_tokens": max_new_tokens, "gen": gen_kwargs, "results": {},
    }
    tf = open(out_dir / "exp_reason_operator_traces.jsonl", "w")
    seen = set()
    for L_req in cfg["L_values"]:
        L = min(L_req, len(pairs_all))
        if L in seen:
            continue
        seen.add(L)
        print(f"=== L={L} ===", flush=True)
        summary = measure_L(model, tok, pairs_all[:L], L, n_iters, op_rng,
                            gen_kwargs, max_new_tokens, chunk_size, preamble,
                            question, enable_thinking, device, pad_id, eos_id,
                            tf, save_cap)
        results["results"][str(L)] = summary
    tf.close()
    with open(out_dir / "exp_reason_operator_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_dir / 'exp_reason_operator_results.json'}", flush=True)


if __name__ == "__main__":
    main()
