#!/usr/bin/env python3
"""Lab 02 - look inside: tokens and next-token probabilities.

    python labs/02_next_token.py "The capital of Egypt is"
    python labs/02_next_token.py --tokens "Tokenization of 12345 strawberries"

An LLM does exactly one thing: given the tokens so far, output a probability
for every possible next token. Everything else (chat, reasoning, tools) is
built from repeating that step. This lab shows the raw distribution and how
temperature reshapes it.

Needs llama-server (uses its /tokenize and /completion endpoints).
"""
import argparse
import math

import llm


def show_tokens(text: str) -> None:
    toks = llm.native("/tokenize", {"content": text, "with_pieces": True})["tokens"]
    print(f"{len(text)} characters -> {len(toks)} tokens:")
    print("  " + " | ".join(repr(t["piece"])[1:-1] for t in toks))
    print("  ids:", [t["id"] for t in toks])
    print("\nTip: try numbers, other languages, code, and emoji - token counts differ a lot.")


def show_next(prompt: str, k: int, steps: int) -> None:
    text = prompt
    print(f"Prompt: {prompt!r}\n")
    for step in range(steps):
        # Raw completion (no chat template) so we see the model's own continuation.
        r = llm.native("/completion", {"prompt": text, "n_predict": 1, "n_probs": k,
                                        "temperature": 0, "cache_prompt": True})
        probs = r["completion_probabilities"][0]["top_logprobs"]
        print(f"step {step + 1}: next-token candidates")
        for cand in probs:
            p = math.exp(cand["logprob"])
            bar = "#" * int(p * 40)
            print(f"   {cand['token']!r:>16}  {p:6.1%}  {bar}")
        best = probs[0]["token"]
        text += best
        print(f"   -> picked {best!r} (greedy)\n")

    # How temperature changes the distribution of the first step
    r = llm.native("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": k, "temperature": 0})
    logps = [c["logprob"] for c in r["completion_probabilities"][0]["top_logprobs"]]
    names = [c["token"] for c in r["completion_probabilities"][0]["top_logprobs"]]
    print("Temperature reshapes the same scores (top candidates only, renormalized):")
    for t in (0.3, 0.7, 1.0, 1.5):
        w = [math.exp(lp / t) for lp in logps]
        z = sum(w)
        cells = ", ".join(f"{n!r}:{x / z:.0%}" for n, x in zip(names[:4], w[:4]))
        print(f"   T={t:<4} {cells}")
    print("\nLow T = predictable, high T = creative/random. T=0 always takes the top token.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="The three most popular programming languages are")
    ap.add_argument("--tokens", action="store_true", help="only show how the text is tokenized")
    ap.add_argument("-k", type=int, default=6, help="candidates to show")
    ap.add_argument("--steps", type=int, default=3, help="tokens to generate step by step")
    a = ap.parse_args()
    if a.tokens:
        show_tokens(a.text)
    else:
        show_tokens(a.text)
        print()
        show_next(a.text, a.k, a.steps)


if __name__ == "__main__":
    main()
