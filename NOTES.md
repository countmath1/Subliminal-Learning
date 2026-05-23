# Notes

Interpretation caveats for anyone reading results — including future-you.

## Interpreting null results on Instruct models

If `exp1` (or any priming experiment on Llama-3.1-8B-Instruct) shows a ≤2pp shift in `P(5 | {5,7})` between control and treatment, **do not conclude subliminal priming fails.**

RLHF training explicitly flattens model behavior on arbitrary-choice tasks ("pick a number between 5 and 7") toward 50/50. A null result on Instruct is consistent with both:

(a) priming genuinely doesn't work, and
(b) priming works but RLHF dominates the small signal.

These need separate evidence to distinguish. Before declaring null, run **exp1c**.

## exp1c candidate

Base Llama-3.1-8B (not Instruct), identical exp1 design. Same 34-item animal preference list, same control prompt. Triggered only if exp1 on Instruct shows ≤2pp shift; otherwise deferred.

- If base also shows null → priming genuinely absent at this list size; revisit list length / category / model family.
- If base shows substantial shift → Instruct result is RLHF-suppressed, not evidence against priming.

Note: base models don't have a chat template. Use the raw prompt directly instead of `tok.apply_chat_template`.
