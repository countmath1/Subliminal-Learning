"""Experiment 1: does an irrelevant preference list shift Llama-3.1-8B-Instruct's
5-vs-7 choice? Runs control (bare question) and treatment (list + question),
measures both via logits at the answer position and via 100x sampling at T=1,
writes JSON to --out/exp1_results.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_messages(prompt: str):
    return [{"role": "user", "content": prompt}]


def assemble_prompt(cond, cfg):
    """Build a prompt from a condition spec.

    String form: used as-is (backward compatibility for old configs).
    Dict form: assembled as `preamble + list_file contents + question`,
    joined by blank lines. Per-condition `preamble` overrides the
    top-level `preamble` in cfg; either is optional (only added if a
    `list_file` is present).
    """
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
    return "\n\n".join(parts)


def candidate_first_tokens(tok, digit: int) -> list[int]:
    """Vocab ids whose decoded string starts with `digit` (after stripping
    leading whitespace). Captures "5", " 5", "5\n", "5.", etc. — every
    single-token way the model could begin its answer with that digit."""
    target = str(digit)
    return [tid for tid in range(len(tok))
            if tok.decode([tid]).lstrip().startswith(target)]


def measure(model, tok, prompt, n_samples, temperature, seed, device, sample_chunk_size=100):
    messages = build_messages(prompt)
    # transformers 5.x: apply_chat_template returns a BatchEncoding by default,
    # not a raw tensor. Extract input_ids explicitly.
    encoded = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(device)

    # All single-token ways the model could start its answer with 5 or 7.
    # Aggregating over these is robust to BPE variants (" 5" vs "5" vs "5\n").
    ids_5 = candidate_first_tokens(tok, 5)
    ids_7 = candidate_first_tokens(tok, 7)
    assert ids_5 and ids_7, "no candidate tokens found for 5 or 7"

    # --- logit measurement: one forward pass ---
    with torch.no_grad():
        out = model(input_ids)
    next_logits = out.logits[0, -1, :].float()
    log_sm = torch.log_softmax(next_logits, dim=-1)
    logp_5 = torch.logsumexp(log_sm[ids_5], dim=0).item()
    logp_7 = torch.logsumexp(log_sm[ids_7], dim=0).item()
    p5_rel = float(np.exp(logp_5) / (np.exp(logp_5) + np.exp(logp_7)))
    top5_idx = int(torch.argmax(log_sm[ids_5]).item())
    top7_idx = int(torch.argmax(log_sm[ids_7]).item())
    top5_id, top7_id = ids_5[top5_idx], ids_7[top7_idx]
    # Raw pre-softmax logits for the single most-likely "5-" and "7-" tokens.
    # Useful for inspecting the model's calibration directly; the difference
    # top_5_raw_logit - top_7_raw_logit equals log_odds_5_over_7 (partition
    # function cancels), but the individual values are informative on their
    # own — e.g., for tracking how absolute confidence shifts across conditions.
    top5_raw_logit = next_logits[top5_id].item()
    top7_raw_logit = next_logits[top7_id].item()

    # --- sampling measurement ---
    torch.manual_seed(seed)
    # Chunked sampling to bound peak KV-cache memory for long prompts.
    # For a 6k-token prompt at batch=1000, the cache would exceed L40S
    # memory; chunking keeps peak memory linear in chunk size.
    # Generation-config overrides (top_p=1.0, top_k=0, repetition_penalty=1.0)
    # match the sampled distribution to the raw next-token distribution
    # the logit measurement reads — without them, Qwen's bundled defaults
    # (top_p=0.8, top_k=20, repetition_penalty=1.05) distort sampling.
    counts = {"5": 0, "7": 0, "other": 0}
    other_examples = []
    remaining = n_samples
    while remaining > 0:
        chunk_n = min(sample_chunk_size, remaining)
        chunk_gen = model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=2,
            num_return_sequences=chunk_n,
            pad_token_id=tok.eos_token_id,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
        )
        new_tokens = chunk_gen[:, input_ids.shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        for t in texts:
            s = t.strip()
            if s.startswith("5"):
                counts["5"] += 1
            elif s.startswith("7"):
                counts["7"] += 1
            else:
                counts["other"] += 1
                if len(other_examples) < 10:
                    other_examples.append(t)
        remaining -= chunk_n
    assert sum(counts.values()) == n_samples

    return {
        "prompt": prompt,
        "logit": {
            "logp_5": logp_5,
            "logp_7": logp_7,
            "p_5_given_5_or_7": p5_rel,
            "log_odds_5_over_7": logp_5 - logp_7,
        },
        "sampling": {
            "n": n_samples,
            "counts": counts,
            "freq_5": counts["5"] / n_samples,
            "freq_7": counts["7"] / n_samples,
            "freq_other": counts["other"] / n_samples,
            "other_examples": other_examples,
        },
        "tokens": {
            "n_candidate_5": len(ids_5),
            "n_candidate_7": len(ids_7),
            "top_5_id": top5_id,
            "top_5_str": tok.decode([top5_id]),
            "top_5_raw_logit": top5_raw_logit,
            "top_7_id": top7_id,
            "top_7_str": tok.decode([top7_id]),
            "top_7_raw_logit": top7_raw_logit,
        },
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
        cfg["model"],
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    results = {
        "model": cfg["model"],
        "n_samples": cfg["n_samples"],
        "temperature": cfg["temperature"],
        "seed": cfg["seed"],
        "conditions": {},
    }
    sample_chunk_size = cfg.get("sample_chunk_size", 100)
    for name, cond in cfg["conditions"].items():
        print(f"--- {name} ---", flush=True)
        prompt = assemble_prompt(cond, cfg)
        r = measure(
            model, tok,
            prompt=prompt,
            n_samples=cfg["n_samples"],
            temperature=cfg["temperature"],
            seed=cfg["seed"],
            device=device,
            sample_chunk_size=sample_chunk_size,
        )
        results["conditions"][name] = r
        print(json.dumps(r["logit"], indent=2), flush=True)
        print(json.dumps(r["sampling"]["counts"]), flush=True)

    out_path = out_dir / "exp1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
