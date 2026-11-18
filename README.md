# modelmania

A hands-on course for running open-weight LLMs **on your own machine** and on
**rented cloud GPUs**: from your first chat to choosing, measuring, serving,
building with and (eventually) fine-tuning models.

**Start here → [ROADMAP.md](ROADMAP.md)** (8 phases, each with goals, exercises and a "done when" checklist).

---

## The short answer: which model, which tool?

**Tool:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (v1.0, 2026) as the
engine you learn on. Ollama and LM Studio are friendly wrappers around it; vLLM
for serving many users on cloud GPUs. They all speak the same OpenAI-compatible
API, so everything in this repo works against any of them.

**Models (Nov 2026), by how much memory you can give the model:**

| Memory budget | Typical machine | Start with |
|---|---|---|
| < 4 GB | old laptop, 4 GB GPU | Qwen3.5-2B |
| 4-8 GB | 8 GB laptop/Mac | Qwen3.5-4B (Q4_0 on a CPU, Q4_K_M on a GPU), Gemma 4 E4B |
| 8-14 GB | 16 GB laptop/Mac, 12 GB GPU | Gemma 4 E4B + Qwen3.5-9B; gpt-oss-20b if ≥ 12 GB free |
| 14-22 GB | **24 GB Mac** (17.8 GiB for the GPU), 16 GB GPU, 32 GB PC | **gpt-oss-20b**; Qwen3.8-27B UD-IQ4_XS if ≥ 16 GiB; Gemma 4 26B-A4B |
| 22-40 GB | 24-32 GB GPU, 36-48 GB Mac | **Qwen3.8-27B** (Q4_K_M), Qwen3.6-35B-A3B, Gemma 4 31B |
| 40 GB+ | 64 GB+ Mac, 2 GPUs, cloud | gpt-oss-120b, Qwen3.5-122B-A10B |

Run `scripts/detect-hardware.sh` to find your budget and tier. Details and
reasoning are in [docs/02-models.md](docs/02-models.md).

**On the machines this repo was measured on** (same llama.cpp commit, same files):

- **MacBook Pro, M4 Pro, 24 GB** (Metal, 273 GB/s, 17.8 GiB usable by the GPU):
  **gpt-oss-20b** is both the most accurate *and* the fastest (24/24 on the mini-eval,
  81 tok/s). **Qwen3.8-27B** (UD-IQ4_XS) is the best non-reasoning and coding model
  (21/24, 14 tok/s). **Gemma 4 E4B** (66 tok/s, vision) and **Qwen3.5-4B** (69 tok/s)
  are the light, fast options. Every model ran at 64-76% of peak bandwidth.
- **Reference baseline: 4-vCPU cloud VM, 15 GiB RAM, no GPU**: the same ranking at
  10-11 tok/s, with Qwen3.5-4B **Q4_0** clearly faster than Q4_K_M (on Metal the gap
  is only 9%). Dense 9B+ models were too slow there.

Full data: [docs/05-measurements.md](docs/05-measurements.md). To add your machine:
`python3 tools/mmbench.py --download`, then `--compare`.

## 10-minute quick start

```bash
scripts/detect-hardware.sh                                   # what can I run?
scripts/build-llama-cpp.sh                                   # or: brew install llama.cpp / llama.app installer
scripts/get-model.sh unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_0.gguf
scripts/serve.sh models/Qwen3.5-4B-Q4_0.gguf --no-think      # web UI + API on http://127.0.0.1:9931
python3 labs/01_chat.py                                      # (second terminal) streaming chat with stats
```

## What's in here

| Path | What |
|---|---|
| [ROADMAP.md](ROADMAP.md) | the learning path, phase by phase |
| [docs/01-how-it-works.md](docs/01-how-it-works.md) | tokens, prefill/decode, memory math, quantization, KV cache, MoE, thinking, bandwidth table |
| [docs/02-models.md](docs/02-models.md) | model families, picks by hardware tier and task, which file to download, trust |
| [docs/03-tools.md](docs/03-tools.md) | llama.cpp / Ollama / LM Studio / MLX / vLLM cheat sheets, troubleshooting |
| [docs/04-cloud.md](docs/04-cloud.md) | renting GPUs: choosing, providers, workflow, security, cost |
| [docs/05-measurements.md](docs/05-measurements.md) | real benchmarks, perplexity, memory and eval results |
| `scripts/` | `detect-hardware`, `build-llama-cpp`, `get-model`, `serve`, `bench` |
| `tools/memcalc.py` | memory + speed predictor that reads a GGUF (matches llama.cpp's KV allocation exactly) |
| `tools/mmbench.py` | one-command benchmark with a pinned model set: fingerprint, speed, % of peak bandwidth, eval; `--compare` across machines |
| `labs/` | 01 chat · 02 next-token probabilities · 03 your own eval · 04 JSON schema · 05 tool calling · 06 RAG |
| `cloud/bootstrap-gpu.sh` | turn a rented GPU box into a private, authenticated endpoint (llama.cpp or vLLM) |
| `results/` | `<machine-id>-<date>.json` from mmbench, plus one folder per machine with bench, perplexity and eval runs |

Labs and mmbench use only the Python standard library (3.9+; tested with macOS's
`/usr/bin/python3` 3.9 and 3.13). `tools/memcalc.py` needs `pip install gguf`.
Model files go in `models/` (git-ignored).
