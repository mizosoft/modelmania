#!/usr/bin/env python3
"""Lab 06 - RAG: let a small model answer questions about YOUR documents.

Needs two servers (two terminals):
    PORT=9932 scripts/serve.sh models/embeddinggemma-2-Q8_0.gguf --embedding   # embeddings
    scripts/serve.sh models/Qwen3.5-4B-Q4_0.gguf                              # chat (9931)

Then:
    python labs/06_rag.py "How much memory does the KV cache use?"
    python labs/06_rag.py --docs path/to/your/notes "your question"

Pipeline (Retrieval-Augmented Generation):
  1. split documents into chunks
  2. embed every chunk -> a vector (list of numbers) that captures its meaning
  3. embed the question, rank chunks by cosine similarity
  4. paste the top chunks into the prompt and ask the model to answer from them
The index is cached in results/rag-index.json so step 2 runs once.
"""
import argparse
import glob
import hashlib
import json
import math
import os

import llm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_URL = os.environ.get("EMBED_BASE_URL", "http://127.0.0.1:9932/v1")


def chunks_from(paths, size=900, overlap=150):
    for p in paths:
        text = open(p, encoding="utf-8", errors="ignore").read()
        # split on blank lines, then pack paragraphs into ~size-character chunks
        paras, buf = [x.strip() for x in text.split("\n\n") if x.strip()], ""
        # hard-split very long paragraphs (big tables) so no chunk outgrows the embedder
        paras = [p[i:i + size] for p in paras for i in range(0, len(p), size)]
        for para in paras:
            if len(buf) + len(para) > size and buf:
                yield {"source": os.path.relpath(p, ROOT), "text": buf}
                buf = buf[-overlap:]
            buf += "\n\n" + para
        if buf.strip():
            yield {"source": os.path.relpath(p, ROOT), "text": buf}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-9)


def build_index(paths):
    chunks = list(chunks_from(paths))
    key = hashlib.sha1(json.dumps([c["text"] for c in chunks]).encode()).hexdigest()
    cache = os.path.join(ROOT, "results", "rag-index.json")
    if os.path.exists(cache):
        idx = json.load(open(cache))
        if idx.get("key") == key:
            return idx["chunks"]
    print(f"Embedding {len(chunks)} chunks (one-time)...")
    for i in range(0, len(chunks), 16):
        batch = chunks[i:i + 16]
        for c, v in zip(batch, llm.embed([c["text"] for c in batch], base=EMBED_URL)):
            c["vec"] = v
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"key": key, "chunks": chunks}, open(cache, "w"))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="Why is gpt-oss-20b fast on a CPU?")
    ap.add_argument("--docs", default=os.path.join(ROOT, "docs"),
                    help="folder of .md/.txt files (default: this repo's docs)")
    ap.add_argument("-k", type=int, default=4, help="chunks to retrieve")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.docs, "**", "*.md"), recursive=True) +
                   glob.glob(os.path.join(a.docs, "**", "*.txt"), recursive=True))
    if os.path.isdir(a.docs) and a.docs.rstrip("/").endswith("docs"):
        paths += [os.path.join(ROOT, "ROADMAP.md"), os.path.join(ROOT, "README.md")]
    paths = [p for p in paths if os.path.exists(p)]
    index = build_index(paths)

    qv = llm.embed([a.question], base=EMBED_URL)[0]
    ranked = sorted(index, key=lambda c: cosine(qv, c["vec"]), reverse=True)[:a.k]
    print(f"Q: {a.question}\n\nRetrieved:")
    for c in ranked:
        print(f"  {cosine(qv, c['vec']):.3f}  {c['source']}: {c['text'].strip()[:80]!r}...")

    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text'].strip()}" for c in ranked)
    msgs = [
        {"role": "system", "content": "Answer ONLY from the provided context. "
                                      "Cite sources in [brackets]. If the answer is not "
                                      "in the context, say you don't know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {a.question}"},
    ]
    print("\nA: ", end="", flush=True)
    for piece in llm.stream_chat(msgs, temperature=0.2, max_tokens=400, **llm.no_think()):
        if "content" in piece:
            print(piece["content"], end="", flush=True)
        elif "stats" in piece:
            s = piece["stats"]
            print(f"\n\n[{s['prompt_tokens']} prompt tokens -> first token after "
                  f"{s['time_to_first_token_s']:.1f}s: long prompts cost prefill time]")


if __name__ == "__main__":
    main()
