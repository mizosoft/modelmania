# Measurements on the reference machine

All numbers below were measured while this repo was written (2026-11-18), so
the advice in the other docs rests on data, not folklore. Re-run them on your
own machine with the commands shown. Your numbers will differ, and *that's the point*.

**Machine:** cloud VM, 4 vCPUs of an Intel Xeon (AVX-512, AVX-512-VNNI/BF16, AMX),
15 GiB RAM, **no GPU**, Ubuntu 24.04.
**Software:** llama.cpp 1.0.0-dev (commit `a300a9c`), built with `-DGGML_NATIVE=ON`, 4 threads.

## 1. Speed (llama-bench, `pp512` / `tg128`, tokens per second)

```bash
scripts/bench.sh models/*.gguf        # or: llama-bench -m <file> -t 4 -p 512 -n 128
```

| Model | Quant | File (GiB) | Prefill t/s | **Generate t/s** |
|---|---|---|---|---|
| Qwen3.5-0.8B | Q4_K_M | 0.50 | 396 | **29.3** |
| Qwen3.5-2B | Q4_K_M | 1.18 | 204 | **15.3** |
| Qwen3.5-4B | Q4_K_M | 2.54 | 80-83 | **7.0-7.5** |
| Qwen3.5-4B | **Q4_0** | 2.40 | 57 | **11.0** |
| Qwen3.5-9B | Q4_K_M | 5.28 | 47 | **4.2** |
| Gemma 4 E4B (official QAT) | Q4_0 | 4.79 | 139 | **11.4** |
| gpt-oss-20b (MoE, 4/32 experts) | MXFP4 | 11.27 | 37 | **10.0** |

What this shows:
- **Decode speed tracks bytes read per token**, not the parameter count. The 21B
  MoE generates faster than the 9B dense model because it reads ~2.4 GiB per token
  instead of ~4.8 GiB (`tools/memcalc.py` computes this from the file).
- **Quant format matters on CPUs.** `Q4_0` has hand-optimized x86 kernels
  (llama.cpp repacks it for AVX-512/AMX at load time): +56% decode vs `Q4_K_M`,
  but slower prefill. On a GPU the gap mostly disappears.
- **Gemma 4 E4B is faster than its file size suggests**: 2.2 GiB of its 4.8 GiB
  are per-layer embeddings that are only looked up, and it uses `Q4_0`.
- **Prefill is 5-14× faster than decode**, so long prompts cost seconds before
  the first word, especially for gpt-oss (37 t/s prefill: a 4,000-token
  document waits ~2 minutes on this machine).
- Effective bandwidth here is ~20-30 GB/s, which is laptop-class. The 4-vCPU
  slice, not the server, is the limit.

## 2. Quality cost of quantization (perplexity, lower = better)

```bash
llama-perplexity -m models/<file>.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 16
```

| Qwen3.5-4B | PPL (wikitext-2, 16×512 tokens) |
|---|---|
| Q4_K_M | 11.07 ± 0.48 |
| Q4_0 | 11.17 ± 0.49 |

That's a +0.9% difference, well inside the error bars. On this CPU, `Q4_0` is the
better trade. (For a sharper test, compare KL-divergence against the BF16 model:
`llama-perplexity --kl-divergence-base` / `--kl-divergence`.)

## 3. Memory: calculator vs llama.cpp's own allocation

```bash
python3 tools/memcalc.py models/<file>.gguf --ctx 4096 32768 131072
llama-server -m models/<file>.gguf -c 32768 --log-verbosity 4 2>&1 | grep llama_kv_cache
```

| Model | Context | memcalc | llama.cpp log |
|---|---|---|---|
| Qwen3.5-9B (8 of 32 layers full attention) | 4,096 | 128 MiB | 128.00 MiB |
| Qwen3.5-9B | 131,072 | 4.00 GiB | 4096.00 MiB (+ 50 MiB fixed recurrent state) |
| Gemma 4 E4B (4 full + 20 sliding-window layers, 18 shared) | 32,768 | 552 MiB | 512 + 40 MiB |
| gpt-oss-20b (12 full + 12 sliding-window layers) | 32,768 | 786 MiB | 768 + 18 MiB |

A classic transformer of Qwen3.5-9B's shape would need 4× the KV of the hybrid
design: 16 GiB at 131K context, more than this whole machine.

Process RSS for Qwen3.5-9B was 9.6 GiB at 4K context, more than the 5.3 GiB
file. That's because llama.cpp copies weights into an AMX-optimized buffer while
the file also stays memory-mapped (page cache, reclaimable). Always measure.

## 4. Answer quality (labs/03_mini_eval.py, 24 auto-checked questions, temperature 0)

```bash
python3 labs/03_mini_eval.py [--think] [--reasoning-effort low]
python3 labs/03_mini_eval.py --compare
```

| Model | Mode | Score | Total time | Notes |
|---|---|---|---|---|
| **gpt-oss-20b** MXFP4 | reasoning effort *low* | **24/24** | 157 s | 16-200 reasoning tokens per question |
| Gemma 4 E4B QAT Q4_0 | thinking off | 18/24 | 20 s | fastest; misses on code and two reasoning items |
| Qwen3.5-9B Q4_K_M | thinking off | 17/24 | 58 s | math 1/4 without thinking |
| Qwen3.5-4B Q4_0 | thinking off | 16/24 | 30 s | |
| Gemma 4 E4B QAT Q4_0 | thinking **on** (math + reasoning only) | 9/10 | 97 s | vs 7/10 thinking off |

This is a tiny eval, so don't over-read it. But it shows the pattern you'll
see everywhere: **reasoning buys accuracy on math and logic, at 5-50× the time**,
and a well-designed MoE reasoning model can be both the most accurate *and*
usable on a CPU.

## 5. Recommendation for this machine (4 vCPU, 15 GiB, no GPU)

| Role | Model | Why |
|---|---|---|
| **Best answers** | `ggml-org/gpt-oss-20b-GGUF` (MXFP4, 12.1 GB) | 24/24, 10 tok/s; needs ~12.4 GiB at 32K context; slow with long prompts |
| **Fast all-rounder + vision** | `google/gemma-4-E4B-it-qat-q4_0-gguf` (5.2 GB) | 11 tok/s, fastest prefill, handles images |
| **Experiments, labs, extraction** | `unsloth/Qwen3.5-4B-GGUF` Q4_0 (2.6 GB) | 11 tok/s, low memory, great with JSON schema |
| Embeddings (RAG) | `ggml-org/embeddinggemma-2-GGUF` Q8_0 (0.3 GB) | small and fast |
| Skip here | dense 9B+ (≤ 4 tok/s), anything 27B+ | needs a GPU, more memory or more cores |
