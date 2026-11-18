"""llm.py - a tiny, dependency-free client for any OpenAI-compatible server.

Works with llama-server, Ollama, LM Studio, vLLM, SGLang, ... because they all
speak the same HTTP API. Reading this file IS part of the course: there is no
magic, just JSON over HTTP.

Configure with environment variables:
    LLM_BASE_URL  default http://127.0.0.1:9931/v1
                  Ollama: http://127.0.0.1:11434/v1   LM Studio: http://127.0.0.1:1234/v1
                  vLLM:   http://127.0.0.1:8000/v1
    LLM_MODEL     model name; llama-server ignores it, Ollama/vLLM need it
    LLM_API_KEY   only if the server was started with an API key
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Iterator

BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:9931/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "local")
API_KEY = os.environ.get("LLM_API_KEY", "none")


def _post(path: str, payload: dict, stream: bool = False, base: str | None = None):
    req = urllib.request.Request(
        (base or BASE_URL) + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:  # server answered, but with an error: show why
        body = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"{base or BASE_URL}{path} -> HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {base or BASE_URL}{path}: {e}\n"
                         "Is the server running? (scripts/serve.sh models/<file>.gguf)")


def no_think() -> dict:
    """Extra request fields that switch off 'thinking' on Qwen3.5 / Gemma 4 templates."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def chat(messages: list[dict], **params) -> dict:
    """One non-streaming chat completion. Returns the full JSON response."""
    payload = {"model": MODEL, "messages": messages, **params}
    with _post("/chat/completions", payload) as r:
        return json.load(r)


def stream_chat(messages: list[dict], **params) -> Iterator[dict]:
    """Stream a chat completion. Yields {'content': str} / {'reasoning': str} pieces,
    then a final {'stats': {...}} item with timing numbers."""
    payload = {"model": MODEL, "messages": messages, "stream": True,
               "stream_options": {"include_usage": True}, **params}
    t0 = time.perf_counter()
    t_first = None
    n_chunks = 0
    usage = None
    with _post("/chat/completions", payload) as r:
        for raw in r:  # Server-Sent Events: lines like "data: {...}"
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                for key, out in (("reasoning_content", "reasoning"), ("content", "content")):
                    if delta.get(key):
                        if t_first is None:
                            t_first = time.perf_counter()
                        n_chunks += 1
                        yield {out: delta[key]}
    t_end = time.perf_counter()
    gen_tokens = (usage or {}).get("completion_tokens", n_chunks)
    gen_time = t_end - (t_first or t_end)
    yield {"stats": {
        "time_to_first_token_s": (t_first - t0) if t_first else None,
        "completion_tokens": gen_tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "tokens_per_s": gen_tokens / gen_time if gen_time > 0 else None,
        "total_s": t_end - t0,
    }}


def embed(texts: list[str], base: str | None = None, model: str | None = None) -> list[list[float]]:
    """Embeddings via /v1/embeddings (llama-server started with --embedding)."""
    with _post("/embeddings", {"model": model or MODEL, "input": texts}, base=base) as r:
        data = json.load(r)["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def server_root() -> str:
    """http://host:port without the /v1 suffix (for llama-server's native endpoints)."""
    return BASE_URL[:-3] if BASE_URL.endswith("/v1") else BASE_URL


def native(path: str, payload: dict) -> dict:
    """Call llama-server's own (non-OpenAI) endpoints, e.g. /tokenize."""
    with _post(path, payload, base=server_root()) as r:
        return json.load(r)
