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

| Memory budget | Start with |
|---|---|
| < 4 GB | Qwen3.5-2B |
| 4-8 GB | Qwen3.5-4B (Q4_0), Gemma 4 E4B |
| 8-14 GB | Gemma 4 E4B + Qwen3.5-9B; gpt-oss-20b if ≥ 12 GB free |
| 14-22 GB | **gpt-oss-20b**, Gemma 4 26B-A4B |
| 22-40 GB | **Qwen3.8-27B**, Qwen3.6-35B-A3B, Gemma 4 31B |
| 40 GB+ | gpt-oss-120b, Qwen3.5-122B-A10B |

Run `scripts/detect-hardware.sh` to find your budget and tier. Details and
reasoning are in [docs/02-models.md](docs/02-models.md).

**On the machine this repo was built and measured on** (4-vCPU cloud VM, 15 GiB RAM,
no GPU): **gpt-oss-20b** gave the best answers (24/24 on the mini-eval, 10 tok/s),
**Gemma 4 E4B** was the fast all-rounder (11 tok/s, vision), and
**Qwen3.5-4B Q4_0** was the lightweight workhorse. Dense 9B+ models were too slow
there. Full data: [docs/05-measurements.md](docs/05-measurements.md).

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
| `labs/` | 01 chat · 02 next-token probabilities · 03 your own eval · 04 JSON schema · 05 tool calling · 06 RAG |
| `cloud/bootstrap-gpu.sh` | turn a rented GPU box into a private, authenticated endpoint (llama.cpp or vLLM) |
| `results/` | saved eval runs (`labs/03_mini_eval.py --compare`) |

Labs use only the Python standard library. `tools/memcalc.py` needs `pip install gguf`.
Model files go in `models/` (git-ignored).
