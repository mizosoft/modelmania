#!/usr/bin/env python3
"""Lab 04 - structured output: make a small model return valid JSON, every time.

    python labs/04_structured.py
    python labs/04_structured.py "Order #5521: 2x USB-C cable ($9.99 each), ship to Lyon by Friday"

Two ways to get JSON:
  1. ask nicely in the prompt   -> usually works, sometimes doesn't
  2. constrained decoding        -> the server only allows tokens that keep the
     output valid against a JSON Schema (llama.cpp turns it into a grammar).
     It cannot produce invalid JSON, even with a tiny model.
"""
import json
import sys

import llm

SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "unit_price": {"type": "number"},
                },
                "required": ["name", "quantity", "unit_price"],
            },
        },
        "destination_city": {"type": "string"},
        "deadline": {"type": ["string", "null"]},
        "urgent": {"type": "boolean"},
    },
    "required": ["order_id", "items", "destination_city", "deadline", "urgent"],
}

DEFAULT = ("Hi! This is about order A-7781. Please send 3 notebooks at 4.50 each and one "
           "fountain pen (25 EUR) to our Porto office. We need it before the 12th - "
           "it's quite urgent, the event can't move.")


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    msgs = [{"role": "system", "content": "Extract the order as JSON."},
            {"role": "user", "content": text}]

    print("Input:", text, "\n")

    print("1) Prompt-only (no constraint):")
    r = llm.chat(msgs, temperature=0, max_tokens=300, **llm.no_think())
    raw = r["choices"][0]["message"]["content"]
    print(raw.strip()[:600])
    try:
        json.loads(raw)
        print("   -> parses as JSON (this time)\n")
    except json.JSONDecodeError:
        print("   -> NOT valid JSON (markdown fences or chatter) - that's the problem\n")

    print("2) Constrained with a JSON Schema:")
    r = llm.chat(msgs, temperature=0, max_tokens=300,
                 response_format={"type": "json_schema",
                                  "json_schema": {"name": "order", "schema": SCHEMA}},
                 **llm.no_think())
    obj = json.loads(r["choices"][0]["message"]["content"])  # guaranteed to parse
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    total = sum(i["quantity"] * i["unit_price"] for i in obj["items"])
    print(f"\n   -> a real Python dict. Order total computed in code: {total:.2f}")
    print("   Caution: the schema guarantees the SHAPE, not the TRUTH. Look at 'deadline' -")
    print("   small models happily invent a year/month that was never in the text.")
    print("   Exercise: add a field (e.g. 'currency' with an enum) and watch it obey.")


if __name__ == "__main__":
    main()
