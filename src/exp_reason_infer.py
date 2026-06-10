"""Reasoning-mode priming experiment for Qwen3 (thinking enabled).

Unlike exp1_infer.py — which reads the 5-vs-7 answer from a single
forward-pass logit — a thinking model's answer is conditioned on the
sampled reasoning trace. There is no exact one-pass shortcut: the answer
distribution is a marginal over reasoning paths, estimable only by
generating full responses and parsing the committed final answer.

This script samples N full responses per condition, parses each, counts,
and saves every trace to a JSONL for auditing the reasoning itself
(e.g. whether the model references the preference list when deciding).
"""
import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def assemble_prompt(cond, cfg):
    """String form -> used verbatim. Dict form -> preamble + list_file + question."""
    if isinstance(cond, str):
        return cond
    parts = []
    preamble = cond.get("preamble") or (cfg.get("preamble") if "list_file" in cond else None)
    if preamble:
        parts.append(preamble.rstrip())
    if "list_file" in cond:
        with open(cond["list_file"]) as f:
            parts.append(f.read().rstrip())
    if "question" in cond:
        parts.append(cond["question"].rstrip())
    return "\n\n".join(parts) + "\n"


def parse_answer(text):
    """Extract the committed 5-vs-7 choice from a thinking-mode response.

    Returns (answer, has_think_close) where answer is '5', '7', or 'other'.

    Strategy, in order:
      1. Drop the <think>...</think> block; work on the answer portion only.
      2. Honor an explicit 'Final answer: N' anchor if present.
      3. Else take the LAST standalone 5 or 7 in the answer portion (the
         final commitment is typically stated last).
      4. Else 'other' (no parseable 5/7 answer).
    """
    has_close = "</think>" in text
    answer_portion = re.sub(r".*</think>", "", text, flags=re.DOTALL) if has_close else text

    m = re.search(r"final answer[:\s]*\**\s*(\d+)", answer_portion, flags=re.IGNORECASE)
    if m:
        d = m.group(1)
        return (d if d in ("5", "7") else "other"), has_close

    standalone = re.findall(r"(?<!\d)([57])(?!\d)", answer_portion)
    if standalone:
        return standalone[-1], has_close

    return "other", has_close


def measure(model, tok, prompt, n_samples, gen_kwargs, seed, device,
            chunk_size, max_new_tokens, save_cap=1000):
    messages = [{"role": "user", "content": prompt}]
    encoded = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attn = encoded["attention_mask"].to(device)
    eos = tok.eos_token_id

    counts = {"5": 0, "7": 0, "other": 0}
    n_truncated = 0
    n_no_think_close = 0
    gen_lens = []
    traces = []

    torch.manual_seed(seed)
    remaining = n_samples
    with torch.inference_mode():
        while remaining > 0:
            cn = min(chunk_size, remaining)
            out = model.generate(
                input_ids,
                attention_mask=attn,
                do_sample=True,
                num_return_sequences=cn,
                max_new_tokens=max_new_tokens,
                pad_token_id=eos,
                **gen_kwargs,
            )
            new = out[:, input_ids.shape[1]:]
            for row in new:
                eos_pos = (row == eos).nonzero()
                truncated = len(eos_pos) == 0
                gen_len = eos_pos[0].item() if len(eos_pos) else row.shape[0]
                gen_lens.append(gen_len)
                if truncated:
                    n_truncated += 1
                text = tok.decode(row, skip_special_tokens=True)
                ans, has_close = parse_answer(text)
                counts[ans] += 1
                if not has_close:
                    n_no_think_close += 1
                if len(traces) < save_cap:
                    traces.append({
                        "answer": ans,
                        "has_think_close": has_close,
                        "truncated": truncated,
                        "gen_len": gen_len,
                        "text": text,
                    })
            remaining -= cn

    n57 = counts["5"] + counts["7"]
    summary = {
        "n": n_samples,
        "counts": counts,
        "freq_5_overall": counts["5"] / n_samples,
        "freq_7_overall": counts["7"] / n_samples,
        "freq_other_overall": counts["other"] / n_samples,
        "p_5_given_5_or_7": (counts["5"] / n57) if n57 else None,
        "n_truncated": n_truncated,
        "n_no_think_close": n_no_think_close,
        "mean_gen_len": sum(gen_lens) / len(gen_lens),
        "max_gen_len": max(gen_lens),
    }
    return summary, traces


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

    gen_kwargs = cfg.get("gen", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    chunk_size = cfg.get("sample_chunk_size", 16)
    max_new_tokens = cfg["max_new_tokens"]

    results = {
        "model": cfg["model"],
        "n_samples": cfg["n_samples"],
        "max_new_tokens": max_new_tokens,
        "gen": gen_kwargs,
        "seed": cfg["seed"],
        "conditions": {},
    }

    trace_path = out_dir / "exp_reason_traces.jsonl"
    with open(trace_path, "w") as tf:
        for name, cond in cfg["conditions"].items():
            print(f"--- {name} ---", flush=True)
            prompt = assemble_prompt(cond, cfg)
            summary, traces = measure(
                model, tok, prompt,
                n_samples=cfg["n_samples"],
                gen_kwargs=gen_kwargs,
                seed=cfg["seed"],
                device=device,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
            )
            results["conditions"][name] = summary
            for t in traces:
                tf.write(json.dumps({"condition": name, **t}) + "\n")
            printable = {k: v for k, v in summary.items() if k != "counts"}
            print(json.dumps(printable, indent=2), flush=True)
            print(json.dumps(summary["counts"]), flush=True)

    out_path = out_dir / "exp_reason_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path} and {trace_path}", flush=True)


if __name__ == "__main__":
    main()
