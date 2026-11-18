#!/usr/bin/env python3
"""Lab 01 - a streaming terminal chat that shows you the numbers.

    python labs/01_chat.py                 # thinking off (fast)
    python labs/01_chat.py --think         # let the model reason first
    python labs/01_chat.py --system "You are a terse pirate."

Commands inside the chat:
    /reset      forget the conversation
    /think      toggle thinking on/off
    /temp 0.2   change temperature
    /history    show how many tokens the conversation uses (the context window fills up!)
    /quit

What to notice:
  * time-to-first-token grows with the length of the conversation (prefill),
  * tokens/s stays roughly constant (decode is memory-bandwidth bound),
  * the whole history is re-sent on every turn: the model itself is stateless.
"""
import argparse
import sys

import llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--think", action="store_true", help="enable thinking/reasoning")
    ap.add_argument("--system", default="You are a helpful, concise assistant.")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    think = args.think
    temp = args.temp
    history = [{"role": "system", "content": args.system}]
    print(f"Connected to {llm.BASE_URL}  (thinking {'on' if think else 'off'}). /quit to exit.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = history[:1]
            print("(conversation cleared)\n")
            continue
        if user == "/think":
            think = not think
            print(f"(thinking {'on' if think else 'off'})\n")
            continue
        if user.startswith("/temp"):
            temp = float(user.split()[1])
            print(f"(temperature = {temp})\n")
            continue
        if user == "/history":
            text = "\n".join(m["content"] for m in history)
            n = len(llm.native("/tokenize", {"content": text})["tokens"])
            print(f"({len(history)} messages, ~{n} tokens of context)\n")
            continue

        history.append({"role": "user", "content": user})
        params = {"temperature": temp, "max_tokens": args.max_tokens}
        if not think:
            params.update(llm.no_think())

        answer = []
        in_reasoning = False
        print("bot> ", end="", flush=True)
        for piece in llm.stream_chat(history, **params):
            if "reasoning" in piece:
                if not in_reasoning:
                    print("\033[2m[thinking] ", end="")  # dim text
                    in_reasoning = True
                print(piece["reasoning"], end="", flush=True)
            elif "content" in piece:
                if in_reasoning:
                    print("\033[0m\n     ", end="")
                    in_reasoning = False
                answer.append(piece["content"])
                print(piece["content"], end="", flush=True)
            else:
                s = piece["stats"]
                ttft = s["time_to_first_token_s"]
                tps = s["tokens_per_s"]
                print("\033[0m")
                print(f"\033[2m     [{s['prompt_tokens']} prompt tok | first token after "
                      f"{ttft:.2f}s | {s['completion_tokens']} tok @ {tps:.1f} tok/s]\033[0m\n"
                      if ttft and tps else "")
        history.append({"role": "assistant", "content": "".join(answer)})


if __name__ == "__main__":
    sys.exit(main())
