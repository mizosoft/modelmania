#!/usr/bin/env python3
"""Lab 03 - your own tiny benchmark. Trust your evals, not leaderboards.

    python labs/03_mini_eval.py                          # labs/evals/basic.jsonl
    python labs/03_mini_eval.py --think --max-tokens 2048
    python labs/03_mini_eval.py --label qwen3.5-4b-q4_0  --set labs/evals/basic.jsonl
    python labs/03_mini_eval.py --only math,reasoning --think
    python labs/03_mini_eval.py --think --reasoning-effort low   # gpt-oss
    python labs/03_mini_eval.py --compare                # table of all saved runs

Each line of the eval file is a question with an automatic checker:
    number | exact | regex | set | word_count | json_fields
Add your OWN questions - the ones that matter for what you want to do.

Results are saved to results/eval-<label>.json so you can compare models.
"""
import argparse
import glob
import json
import os
import re
import statistics
import time
import unicodedata
import urllib.request

import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def clean(text: str) -> str:
    text = text or ""
    # drop any leaked <think>...</think> and markdown decoration
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("**", "").replace("`", "").strip()
    return text


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    return s.strip(" .!\"'")


def check(item: dict, raw: str) -> bool:
    out = clean(raw)
    kind, exp = item["check"], item["a"]
    if kind == "number":
        nums = re.findall(r"-?\d+(?:\.\d+)?", out.replace(",", ""))
        return bool(nums) and abs(float(nums[-1]) - float(exp)) < 1e-6
    if kind == "exact":
        return norm(out) == exp
    if kind == "regex":
        return re.search(exp, norm(out), flags=re.I) is not None
    if kind == "set":
        got = {norm(x) for x in re.split(r"[,\n]", out) if x.strip()}
        return got == set(exp)
    if kind == "word_count":
        return len(re.findall(r"[A-Za-z']+", out)) == exp
    if kind == "json_fields":
        m = re.search(r"\{.*\}", out, flags=re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return False
        return all(norm(str(obj.get(k, ""))) == v for k, v in exp.items())
    raise ValueError(f"unknown check {kind}")


def model_label() -> str:
    """Ask llama-server which file it loaded (GET /props); fall back to LLM_MODEL."""
    try:
        with urllib.request.urlopen(llm.server_root() + "/props", timeout=5) as r:
            props = json.load(r)
        return os.path.basename(props.get("model_path", "model")).replace(".gguf", "")
    except Exception:
        return llm.MODEL


def run(args) -> None:
    items = [json.loads(l) for l in open(args.set) if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        items = [it for it in items if it["cat"] in keep or it["id"] in keep]
    label = args.label or model_label()
    if args.think:
        label += "+think"
    if args.reasoning_effort:
        label += f"-{args.reasoning_effort}"
    if args.only:
        label += "@" + args.only.replace(",", "+")
    print(f"Evaluating '{label}' on {len(items)} questions via {llm.BASE_URL}\n")
    params = {"temperature": 0, "max_tokens": args.max_tokens}
    if not args.think:
        params.update(llm.no_think())
    if args.reasoning_effort:  # gpt-oss style models: low | medium | high
        params["reasoning_effort"] = args.reasoning_effort

    rows = []
    for it in items:
        t0 = time.perf_counter()
        r = llm.chat([{"role": "user", "content": it["q"]}], **params)
        dt = time.perf_counter() - t0
        msg = r["choices"][0]["message"]
        ans = msg.get("content") or ""
        ok = check(it, ans)
        toks = r.get("usage", {}).get("completion_tokens", 0)
        rows.append({"id": it["id"], "cat": it["cat"], "ok": ok, "answer": clean(ans)[:200],
                     "seconds": round(dt, 2), "completion_tokens": toks})
        mark = "PASS" if ok else "fail"
        print(f"  [{mark}] {it['id']:<10} {dt:5.1f}s {toks:4d} tok  -> {clean(ans)[:60]!r}")

    score = sum(r["ok"] for r in rows)
    cats = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r["ok"])
    print(f"\nScore: {score}/{len(rows)}  |  median latency "
          f"{statistics.median(r['seconds'] for r in rows):.1f}s  |  total "
          f"{sum(r['seconds'] for r in rows):.0f}s")
    print("  " + "  ".join(f"{c}: {sum(v)}/{len(v)}" for c, v in cats.items()))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", f"eval-{label}.json")
    json.dump({"label": label, "set": os.path.basename(args.set), "score": score,
               "n": len(rows), "think": args.think, "rows": rows,
               "date": time.strftime("%Y-%m-%d")}, open(path, "w"), indent=1)
    print(f"\nSaved {path}")


def compare() -> None:
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, "results", "eval-*.json")))]
    if not runs:
        print("No saved runs yet.")
        return
    cats = sorted({r["cat"] for run in runs for r in run["rows"]})
    print(f"| model | score | {' | '.join(cats)} | total time |")
    print("|---|---|" + "---|" * len(cats) + "---|")
    for run in sorted(runs, key=lambda r: -r["score"]):
        per = []
        for c in cats:
            v = [r["ok"] for r in run["rows"] if r["cat"] == c]
            per.append(f"{sum(v)}/{len(v)}" if v else "-")
        t = sum(r["seconds"] for r in run["rows"])
        print(f"| {run['label']} | **{run['score']}/{run['n']}** | {' | '.join(per)} | {t:.0f}s |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=os.path.join(ROOT, "labs", "evals", "basic.jsonl"))
    ap.add_argument("--label", help="name for this run (default: model file name)")
    ap.add_argument("--think", action="store_true", help="allow thinking (slower, often more accurate)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--only", help="comma-separated categories or ids, e.g. math,code-3")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="for gpt-oss style models (they always reason)")
    ap.add_argument("--compare", action="store_true", help="print a table of saved runs")
    a = ap.parse_args()
    compare() if a.compare else run(a)


if __name__ == "__main__":
    main()
