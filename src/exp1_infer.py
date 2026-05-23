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


def measure(model, tok, prompt, n_samples, temperature, seed, device):
    messages = build_messages(prompt)
    input_ids = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    # Single-token ids for " 5" and " 7". Asserted, not assumed.
    ids_5 = tok.encode(" 5", add_special_tokens=False)
    ids_7 = tok.encode(" 7", add_special_tokens=False)
    assert len(ids_5) == 1, f'" 5" tokenized to {ids_5}, expected 1 token'
    assert len(ids_7) == 1, f'" 7" tokenized to {ids_7}, expected 1 token'
    tok_5, tok_7 = ids_5[0], ids_7[0]

    # --- logit measurement: one forward pass ---
    with torch.no_grad():
        out = model(input_ids)
    next_logits = out.logits[0, -1, :].float()
    logit_5 = next_logits[tok_5].item()
    logit_7 = next_logits[tok_7].item()
    log_sm = torch.log_softmax(next_logits, dim=-1)
    logp_5 = log_sm[tok_5].item()
    logp_7 = log_sm[tok_7].item()
    p5_rel = float(np.exp(logp_5) / (np.exp(logp_5) + np.exp(logp_7)))

    # --- sampling measurement ---
    torch.manual_seed(seed)
    gen = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=2,
        num_return_sequences=n_samples,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = gen[:, input_ids.shape[1]:]
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

    counts = {"5": 0, "7": 0, "other": 0}
    other_examples = []
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
    assert sum(counts.values()) == n_samples

    return {
        "prompt": prompt,
        "logit": {
            "logit_5": logit_5,
            "logit_7": logit_7,
            "logp_5": logp_5,
            "logp_7": logp_7,
            "p_5_given_5_or_7": p5_rel,
            "logit_diff_5_minus_7": logit_5 - logit_7,
        },
        "sampling": {
            "n": n_samples,
            "counts": counts,
            "freq_5": counts["5"] / n_samples,
            "freq_7": counts["7"] / n_samples,
            "freq_other": counts["other"] / n_samples,
            "other_examples": other_examples,
        },
        "tokens": {"id_space_5": tok_5, "id_space_7": tok_7},
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
        torch_dtype=torch.bfloat16,
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
    for name, key in [("control", "control_prompt"), ("treatment", "treatment_prompt")]:
        print(f"--- {name} ---", flush=True)
        r = measure(
            model, tok,
            prompt=cfg[key],
            n_samples=cfg["n_samples"],
            temperature=cfg["temperature"],
            seed=cfg["seed"],
            device=device,
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
