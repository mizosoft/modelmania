# Choosing models (state of play: November 2026)

Models move fast. This page gives you **today's shortlist** and, more importantly,
**the method** to re-derive it whenever something new ships. Sizes are real file
sizes from Hugging Face; licenses are from the repos' metadata (all checked 2026-11-18).

## The families worth knowing

| Family | Released | Sizes (★ = good local picks) | Notes | License |
|---|---|---|---|---|
| **Qwen3.5** (Alibaba) | Feb 2026 | 0.8B★, 2B★, 4B★, 9B★, 27B, 35B-A3B, 122B-A10B | hybrid linear attention (small KV cache), vision built in, 262K context, thinking on/off | Apache-2.0 |
| **Qwen3.6** | Apr 2026 | 27B★, 35B-A3B★ (MoE) | refresh of the mid sizes; 35B-A3B is a great MoE for 24-32 GB | Apache-2.0 |
| **Qwen3.8** | Aug 2026 | 27B★, Flash-Next (180B MoE) | the current go-to "27B" for 24 GB+ | Apache-2.0 / other |
| **Gemma 4** (Google) | Mar-Jun 2026 | E2B★, E4B★, 12B★, 26B-A4B★ (MoE), 31B | official **QAT Q4_0** GGUFs; vision; "E" = effective size, runs faster than its file size suggests | Apache-2.0 |
| **gpt-oss** (OpenAI) | Aug 2025 | 20b★ (MoE, 3.6B active), 120b (MoE) | always-on reasoning with low/medium/high effort; ships in MXFP4 | Apache-2.0 |
| **Mistral** | 2025-26 | Devstral Small 2 24B (coding), Mistral Small 3.2 24B, Small 4 119B | strong coding/agent models | Apache-2.0 |
| **Phi-4** (Microsoft) | 2025-26 | mini, reasoning, reasoning-vision 15B | small and efficient; best for narrow tasks | MIT |
| **Embeddings** | | embeddinggemma-2 (0.3 GB★), Qwen3-Embedding 0.6B/4B/8B, Qwen3-Reranker | for search and RAG, not chat | Apache-2.0 |
| **Speech** | | Qwen3-ASR 1.7B, Whisper large-v3-turbo; Qwen3-TTS | speech to text / text to speech | Apache-2.0 / MIT |

Older names you'll still see in tutorials (Llama 3.x, Qwen2.5, Mistral 7B, Gemma 3)
still work, but have been overtaken at every size. Use them only for comparison.

## By hardware tier

The **memory budget** is VRAM (GPU), about ⅔-¾ of RAM (Apple Silicon unified
memory: measured 17.8 GiB = 74% on a 24 GB M4 Pro with macOS 26), or RAM minus ~4 GB
(CPU-only). `scripts/detect-hardware.sh` computes it, asking llama.cpp for the exact Metal figure.
Leave room for the KV cache. `tools/memcalc.py` tells you exactly how much.

| Tier | Budget | Typical machine | Start with | Also try |
|---|---|---|---|---|
| **T0** | < 4 GB | old laptop, Raspberry Pi 5, 4 GB GPU | Qwen3.5-2B Q4_K_M (1.3 GB) | Qwen3.5-0.8B (0.5 GB) |
| **T1** | 4-8 GB | 8 GB laptop / Mac, 6-8 GB GPU | Qwen3.5-4B Q4_0 (2.6 GB) | Gemma 4 E4B QAT (5.2 GB), Gemma 4 E2B (3 GB) |
| **T2** | 8-14 GB | 16 GB laptop/Mac, 12 GB GPU | Gemma 4 E4B QAT (5.2 GB) + Qwen3.5-9B Q4_K_M (5.7 GB) | Gemma 4 12B QAT (7 GB); gpt-oss-20b (12.1 GB) if ≥ 12 GB free |
| **T3** | 14-22 GB | 16 GB GPU, 24 GB Mac, 32 GB RAM PC | **gpt-oss-20b** (12.1 GB) | Qwen3.8-27B UD-IQ4_XS (14.4 GB, if ≥ 16 GiB), Gemma 4 26B-A4B MXFP4 (16.6 GB, tight), Qwen3.5-9B Q8_0 |
| **T4** | 22-40 GB | RTX 3090/4090/5090, 36-48 GB Mac, 64 GB RAM PC | **Qwen3.8-27B** Q4_K_M (16.6 GB) | Qwen3.6-35B-A3B (22 GB), Gemma 4 31B Q4 (18 GB), Devstral Small 2 |
| **T5** | 40+ GB | 64-128 GB Mac, Strix Halo, DGX Spark, 2 GPUs, cloud 80 GB | gpt-oss-120b (63 GB) | Qwen3.5-122B-A10B (75 GB), 27B-35B models at Q8 |

**CPU-only or modest-bandwidth machines: prefer MoE** (gpt-oss-20b, Gemma 4 26B-A4B,
Qwen3.6-35B-A3B). They read only 3-4B parameters per token. That holds on a Mac too: on
an M4 Pro, gpt-oss-20b generated 81 tok/s, against 14 for the dense Qwen3.8-27B. The dense 27B was
still the better non-reasoning model (21/24 vs 18/24 for the 4B models), and 14 tok/s is
fine for reading. See [docs/05](05-measurements.md).
**Discrete GPU: prefer models that fit entirely in VRAM.** Spilling to system RAM
cuts speed by roughly the ratio of the two bandwidths. The exception is MoE with
`--n-cpu-moe`.

## By task

| Task | Small machine | Big machine |
|---|---|---|
| General chat / writing | Gemma 4 E4B, Qwen3.5-4B (thinking off) | Qwen3.8-27B, Gemma 4 31B |
| Reasoning / math | gpt-oss-20b (effort low/medium) | gpt-oss-120b, Qwen3.8-27B (thinking on) |
| Coding | Qwen3.5-9B, gpt-oss-20b | Qwen3.8-27B, Qwen3.6-35B-A3B, Devstral Small 2, Qwen3-Coder-30B-A3B |
| Extraction / classification / JSON | Qwen3.5-2B/4B + JSON schema (lab 04) | same; small is fine here |
| Vision (images, screenshots, OCR) | Gemma 4 E4B, Qwen3.5-4B (+ `mmproj`) | Gemma 4 31B, Qwen3.8-27B |
| RAG embeddings | embeddinggemma-2 | Qwen3-Embedding-4B/8B + Qwen3-Reranker |
| Agents / tool calling | Qwen3.5-9B, gpt-oss-20b | Qwen3.8-27B, Qwen3.6-35B-A3B |

## Which file? (quant picking)

1. Find a GGUF repo: the **official** one (Google ships `*-qat-q4_0-gguf`,
   Qwen ships some `-GGUF` repos), **ggml-org** (made by the llama.cpp team),
   **unsloth** or **bartowski**. These are the reliable quantizers.
2. Start at **Q4_K_M** (or the official **QAT Q4_0**). **CPU-only**: also try
   **Q4_0**, which has the fastest kernels. On the reference VM it was 56% faster
   with the same perplexity. **GPU / Apple Metal**: the kernel advantage disappears.
   On an M4 Pro, Q4_0 was only 9% faster (its file is 6% smaller), so choose by quality and size.
3. If it fits with room to spare, step up to Q5_K_M, Q6_K or Q8_0. If it doesn't
   fit, try IQ4_XS or a smaller model. Don't go below Q3 unless the model is huge (70B+).
4. Vision? Also download the repo's `mmproj-*.gguf` (`llama serve -hf` does this
   automatically).
5. "UD" (unsloth dynamic) quants keep important layers at higher precision.
   They're usually a small win at the same size.

`scripts/get-model.sh --list <repo>` shows every file and its size.

## Trust and hygiene

- **Provenance over popularity.** Hugging Face is full of merges and fine-tunes
  with names like `Qwen3.8-27B-TURBO-...-Uncensored-NEO-CODER-MAX`, and some have
  huge download counts. Start from official or well-known quantizer repos.
  Treat everything else as an experiment.
- **GGUF files are data, not code**, which makes them much safer than pickled
  PyTorch checkpoints. But keep your runtime updated: parser bugs do happen.
  Avoid `trust_remote_code=True` with transformers/vLLM unless you've read the code.
- **Licenses**: everything in the shortlist above is Apache-2.0 or MIT. Check
  again for any other model before commercial use.
- **Benchmarks lie by omission.** Leaderboard scores use different prompts,
  sampling and thinking budgets. Your own eval (`labs/03_mini_eval.py` + your
  questions) on your own settings is the only number that counts.

## Reading a model card (checklist)

- [ ] architecture (dense, MoE, hybrid), total *and* active parameters
- [ ] native context length; whether long context needs YaRN
- [ ] thinking mode: default on or off, and how to toggle it
- [ ] **recommended sampling parameters** (copy them!)
- [ ] chat template quirks, tool-call format, vision support
- [ ] license and any usage policy

## Re-deriving this page in the future

```bash
# most-downloaded recent models from an org
curl -s "https://huggingface.co/api/models?author=Qwen&sort=downloads&limit=20" | python3 -m json.tool | grep '"id"'
# all GGUF files + sizes in a repo
scripts/get-model.sh --list unsloth/<Model>-GGUF
# exact memory/speed prediction once downloaded (before: params × bits ÷ 8, see docs/01)
python3 tools/memcalc.py models/<file>.gguf --bandwidth <your GB/s> --budget <your GiB>
```
Then run your eval. That's the whole method.
