#!/usr/bin/env python3
"""Lab 05 - tool calling: the model decides to call YOUR functions.

    python labs/05_tools.py
    python labs/05_tools.py "How many days until 2027-01-01, and what's 2**20?"

How it works (this loop is the heart of every "agent"):
  1. send the conversation + a list of tool descriptions (JSON Schema)
  2. the model either answers, or replies with tool_calls: name + JSON arguments
  3. YOUR code runs the function and appends the result as a 'tool' message
  4. repeat until the model answers in plain text

The model never executes anything itself - you stay in control of what runs.
"""
import ast
import datetime as dt
import json
import operator
import sys

import llm


# --- the tools: ordinary Python functions ----------------------------------
def calculator(expression: str) -> str:
    """Safely evaluate + - * / ** % on numbers (no eval of arbitrary code)."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.USub: operator.neg}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return str(ev(ast.parse(expression, mode="eval").body))


def days_until(date: str) -> str:
    """Days from today until an ISO date (YYYY-MM-DD)."""
    target = dt.date.fromisoformat(date)
    return str((target - dt.date.today()).days)


TOOLS = {"calculator": calculator, "days_until": days_until}
SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for any math.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string",
                                                     "description": "e.g. '(17*23)+4'"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "days_until",
        "description": "Number of days from today until the given date.",
        "parameters": {"type": "object",
                       "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
                       "required": ["date"]}}},
]


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "What is 48729 * 391, and how many days are left until 2027-03-01?"
    messages = [
        {"role": "system", "content": f"Today is {dt.date.today()}. Use tools for math and dates."},
        {"role": "user", "content": question},
    ]
    print("User:", question, "\n")

    for step in range(6):  # never loop forever
        r = llm.chat(messages, tools=SCHEMAS, temperature=0, max_tokens=512, **llm.no_think())
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            print("Assistant:", (msg.get("content") or "").strip())
            return
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"] or "{}")
            try:
                result = TOOLS[name](**args)
            except Exception as e:  # report errors back to the model, don't crash
                result = f"error: {e}"
            print(f"  [step {step + 1}] model called {name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": c.get("id", name),
                             "content": result})
    print("Stopped after too many tool steps.")


if __name__ == "__main__":
    main()
