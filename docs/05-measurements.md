# Measurements

Real numbers from two machines, so the advice in the other docs rests on data,
not folklore. **Run the same thing on your machine** and add it to the table:

```bash
python3 tools/mmbench.py --download    # pinned models, fixed settings, ~10-30 min
python3 tools/mmbench.py --compare     # your machine next to these two
```

Both machines ran the same llama.cpp commit (`a300a9c`, 1.0.0-dev,
`LLAMA_CPP_REF=a300a9c33462f23d9d61e951650d249486f2b0f6 scripts/build-llama-cpp.sh`)
and the same model files, so the differences come from the hardware.

| | **M4 Pro laptop** (primary) | **Cloud VM** (reference baseline) |
|---|---|---|
| id in `results/` | `apple-m4-pro-20gpu-24gb-metal` | `cloud-xeon-4vcpu-15gb-cpu` |
| Measured | 2026-11-18/19 | 2026-11-18 |
| CPU | Apple M4 Pro, 14 cores (10 performance + 4 efficiency) | 4 vCPUs of an Intel Xeon (AVX-512, VNNI/BF16, AMX) |
| GPU | 20-core Apple GPU, **Metal** | none |
| Memory | 24 GB unified; **17.76 GiB usable by the GPU** (llama.cpp's `recommendedMaxWorkingSetSize`) | 15 GiB RAM |
| Peak bandwidth | **273 GB/s** (Apple spec sheet) | unknown (vCPU slice); ~26-32 GB/s achieved |
| OS / build | macOS 26.6, Metal (default), threads 10 | Ubuntu 24.04, `-DGGML_NATIVE=ON`, threads 4 |

## 1. Speed (llama-bench `pp512` / `tg128`, tokens per second)

```bash
scripts/bench.sh models/*.gguf            # or: llama-bench -m <file> -p 512 -n 128
```

| Model | Quant | File GiB | Read/token GiB | M4 Pro prefill | **M4 Pro generate** | % of 273 GB/s | Cloud VM prefill | **Cloud VM generate** |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | Q4_K_M | 0.50 | | | | | 396 | **29.3** |
| Qwen3.5-2B | Q4_K_M | 1.18 | | | | | 204 | **15.3** |
| Qwen3.5-4B | Q4_K_M | 2.54 | 2.54 | 785 | **63.2** | 64% | 80-83 | **7.0-7.5** |
| Qwen3.5-4B | **Q4_0** | 2.40 | 2.40 | 831 | **69.1** | 65% | 57 | **11.0** |
| Qwen3.5-9B | Q4_K_M | 5.28 | | | | | 47 | **4.2** |
| Gemma 4 E4B (official QAT) | Q4_0 | 4.79 | 2.63 | 1200-1335 | **66.1** | 68% | 139 | **11.4** |
| gpt-oss-20b (MoE, 4/32 experts) | MXFP4 | 11.27 | 2.40 | 926 | **80.7** | 76% | 37 | **10.0** |
| Qwen3.8-27B (dense, hybrid attention) | UD-IQ4_XS | 13.36 | 12.52 | 125 | **13.7** | 68% | - | - |

*Read/token* is from `tools/memcalc.py`; *% of peak* = generate t/s × read/token ÷ 273 GB/s.
Blank cells: model not run on that machine. Repeats agree within 1-2% on
generation; prefill varied by up to 10% between runs (Gemma: 1204-1335), likely
thermal, since the laptop had been under load for an hour.

What this shows:
- **Decode speed is bandwidth ÷ bytes read per token, on both machines.** On the
  M4 Pro every model lands at **64-76% of the spec-sheet bandwidth**, from a 4B to
  a 27B. Once you know that one number for a machine, you can predict any model
  on it within ~10%.
- **MoE still wins on a Mac.** gpt-oss-20b (11.3 GiB file) is the *fastest*
  generator here, faster than the 2.4 GiB Qwen3.5-4B, because it reads the same
  ~2.4 GiB per token and its MXFP4 kernels reach the best efficiency (76%).
  Qwen3.8-27B is about the same size on disk but dense: it reads 12.5 GiB per token,
  so it runs 6× slower.
- **Q4_0 vs Q4_K_M depends on the backend.** On the 4-core CPU, `Q4_0` was 56%
  faster (repacked AVX-512/AMX kernels, cheap dequantization). On Metal it's 9%
  faster, which is what its 6% smaller file predicts, and its prefill is faster
  too (831 vs 785; on the CPU Q4_K_M had the faster prefill). On a GPU, choose the
  quant by quality and size, not by kernel speed.
- **Prefill on Metal is 9-20× decode.** On this machine that means ~0.5 s for a
  500-token prompt with the small models and ~4 s with the 27B. On the CPU VM, gpt-oss
  needed ~2 minutes for a 4,000-token document. The Mac needs ~5 s.
- **Threads don't matter with full GPU offload**: `-t 4` and `-t 10` gave 69.5
  and 68.9 t/s. Every layer was on the GPU by default (`-ngl auto`: 33/33 for
  Qwen3.5-4B, 66/66 for Qwen3.8-27B).
- **The M4 Pro is 6-8× faster at generation and 10-25× faster at prefill** than the
  4-vCPU VM. Generation tracks the effective bandwidth (178-208 vs 26-32 GB/s, 6-8×);
  prefill tracks compute, where a 20-core GPU beats 4 CPU cores by more.

## 2. Quality cost of quantization (perplexity, lower = better)

```bash
llama-perplexity -m models/<file>.gguf -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 16
```

| Qwen3.5-4B | M4 Pro (Metal) | Cloud VM (CPU) |
|---|---|---|
| Q4_K_M | 11.10 ± 0.48 | 11.07 ± 0.48 |
| Q4_0 | 11.19 ± 0.49 | 11.17 ± 0.49 |

The two quants differ by +0.8-0.9%, well inside the error bars, and the two
backends agree within 0.03. On this Mac the trade-off is "9% faster" vs "0.8%
lower perplexity": either choice is fine. (For a sharper test, compare
KL-divergence against the BF16 model: `--kl-divergence-base` / `--kl-divergence`.)
16 chunks take ~13 s on the M4 Pro.

## 3. Memory: calculator vs llama.cpp's own allocation

```bash
python3 tools/memcalc.py models/<file>.gguf --ctx 4096 32768 131072
llama-server -m models/<file>.gguf -c 32768 --log-verbosity 4 2>&1 | grep llama_kv_cache
```

| Model | Context | memcalc KV | llama.cpp log (M4 Pro, Metal) | Process RSS (M4 Pro) |
|---|---|---|---|---|
| Qwen3.5-4B Q4_0 (8 of 32 layers full attention) | 32,768 | 1.00 GiB | 1024.00 MiB (+ 50 MiB recurrent state) | 3.6 GiB |
| Qwen3.5-9B Q4_K_M | 131,072 | 4.00 GiB | 4096.00 MiB (+ 50 MiB) *(cloud VM)* | |
| Gemma 4 E4B (4 full + 20 sliding-window layers, 18 shared) | 32,768 | 552 MiB | 512 + 40 MiB | 5.5 GiB |
| gpt-oss-20b (12 full + 12 sliding-window layers) | 32,768 | 786 MiB | 768 + 18 MiB | 11.8 GiB |
| Qwen3.8-27B (16 of 64 layers full attention, + 1 MTP layer) | 32,768 | 2.00 GiB | 2048.00 MiB (+ 150 MiB recurrent state) | 14.0 GiB |

memcalc's KV numbers match llama.cpp exactly on both backends. Its total is a
little low on Metal: compute buffers were 220-360 MiB, more than the 0.3 GiB it
allows for all overhead. Budget ~0.5 GiB of overhead on a Mac.

Two platform differences to know:
- **On the Mac, weights are not copied.** RSS ≈ file + KV + compute buffers, because
  Metal uses the memory-mapped file directly (`MTL0_Mapped`). On the x86 VM, RSS
  for Qwen3.5-9B was 9.6 GiB for a 5.3 GiB file, because llama.cpp repacks the weights
  into an AMX-friendly copy while the file stays mapped.
- **The GPU can't use all of unified memory.** macOS lets Metal wire
  17.76 GiB of this 24 GB machine (74%; the old rule of thumb said ⅔ = 16 GiB).
  Qwen3.8-27B at IQ4_XS + 32K context (15.7 GiB) fits; the Q4_K_M file (15.4 GiB)
  would leave almost no room for context. You can raise the limit until reboot with
  `sudo sysctl iogpu.wired_limit_mb=<MiB>` (at the cost of memory for everything else).

memcalc now also skips multi-token-prediction ("nextn") layers when counting bytes
per token: Qwen3.8-27B ships one (335 MiB) that llama.cpp loads but only runs for
MTP speculative decoding.

## 4. Answer quality (labs/03_mini_eval.py, 24 auto-checked questions, temperature 0)

```bash
python3 labs/03_mini_eval.py [--think] [--reasoning-effort low] --results-dir results/<machine>
python3 labs/03_mini_eval.py --compare
```

| Model | Mode | M4 Pro score | M4 Pro time | Cloud VM score | Cloud VM time |
|---|---|---|---|---|---|
| **gpt-oss-20b** MXFP4 | reasoning effort *low* | **24/24** | 17 s | **24/24** | 157 s |
| **Qwen3.8-27B** UD-IQ4_XS | thinking off | **21/24** | 19 s | - | - |
| Gemma 4 E4B QAT Q4_0 | thinking off | 18/24 | 3 s | 18/24 | 20 s |
| Qwen3.5-4B Q4_0 | thinking off | 18/24 | 4 s | 16/24 | 30 s |
| Qwen3.5-9B Q4_K_M | thinking off | | | 17/24 | 58 s |
| Gemma 4 E4B QAT Q4_0 | thinking **on** (math + reasoning only) | | | 9/10 | 97 s |

**The same file at temperature 0 does not give identical answers on different
backends.** Qwen3.5-4B Q4_0 scored 16/24 on the x86 CPU, 17/24 on an ARM CPU (a Linux
container on the same Mac) and 18/24 on Metal. Two answers flipped (math-3: 145 vs
155 minutes; instr-2: "anaban" vs "ananab"), because each backend sums the numbers in a different order and close
calls between two tokens tip the other way. On the same machine, reruns were
identical (95 of 96 answers matched word for word). So read a ±2 difference on this
24-question eval as noise, and use more questions before you conclude anything.

Qwen3.8-27B is the best non-reasoning model here (code 3/3, where the 4B models get
1/3). gpt-oss-20b is still the most accurate, and on the Mac its reasoning costs
only 17 s for all 24 questions, against 157 s on the VM.

## 5. Predicted before downloading vs measured

This is the Phase 2 skill: `params × bits ÷ 8 + KV` for memory and
`bandwidth ÷ bytes per token` for speed, at 273 GB/s.

| Model | Predicted memory @16K | memcalc @16K | Predicted generate (60-80% of ceiling) | Measured |
|---|---|---|---|---|
| Qwen3.5-4B Q4_0 | 3.2 GiB | 3.21 GiB | 65-85 t/s | **69.1** |
| Gemma 4 E4B Q4_0 | 5.4 GiB | 5.39 GiB | 60-78 t/s | **66.1** |
| gpt-oss-20b MXFP4 | 11.9 GiB | 11.97 GiB | 60-85 t/s | **80.7** |
| Qwen3.8-27B UD-IQ4_XS | 14.7 GiB | 14.67 GiB | 12-16 t/s | **13.7** |

All four predictions were right. A narrower rule for this machine: **~70% of 273 GB/s ≈ 190 GB/s ÷
bytes per token**.

## 6. Recommendations

**M4 Pro, 24 GB (Metal), what to run:**

| Role | Model | Why |
|---|---|---|
| **Default / best answers** | `ggml-org/gpt-oss-20b-GGUF` MXFP4 (12.1 GB) | 24/24, **81 tok/s**, the fastest model here too; ~12 GiB at 32K context |
| **Best non-reasoning, coding** | `unsloth/Qwen3.8-27B-GGUF` UD-IQ4_XS (14.4 GB) | 21/24, code 3/3; 14 tok/s is a comfortable reading speed; ≤ 32K context in the 17.8 GiB GPU limit |
| **Fast + vision** | `google/gemma-4-E4B-it-qat-q4_0-gguf` (5.2 GB) | 66 tok/s, fastest prefill (1,300 t/s), handles images, leaves 12 GiB free for other apps |
| **Labs, extraction, JSON** | `unsloth/Qwen3.5-4B-GGUF` Q4_0 or Q4_K_M (2.6-2.7 GB) | ~65-70 tok/s, tiny; either quant is fine on Metal |
| Doesn't fit well | anything > ~15 GiB file (Qwen3.8-27B Q4_K_M, Qwen3.6-35B-A3B Q4) | over the GPU limit once you add context; needs `iogpu.wired_limit_mb` or a 36 GB+ Mac |

**Cloud VM, 4 vCPU, 15 GiB, no GPU** (reference baseline):

| Role | Model | Why |
|---|---|---|
| **Best answers** | gpt-oss-20b MXFP4 | 24/24, 10 tok/s; slow with long prompts (37 t/s prefill) |
| **Fast all-rounder + vision** | Gemma 4 E4B QAT Q4_0 | 11 tok/s, fastest prefill there |
| **Experiments, labs, extraction** | Qwen3.5-4B **Q4_0** | 11 tok/s (Q4_K_M only 7) |
| Skip there | dense 9B+ (≤ 4 tok/s), anything 27B+ | needs a GPU, more memory or more cores |

Embeddings for RAG, on either machine: `ggml-org/embeddinggemma-2-GGUF` Q8_0 (0.3 GB).

## 7. mmbench: comparable results from any machine

`tools/mmbench.py` turns all of the above into one command. It uses only the Python standard library;
`pip install gguf` lets it recompute bytes per token from each file (otherwise it uses pinned values).

- **Fingerprint**: OS, CPU, cores (and container limits), RAM, GPU and its usable
  memory (via `llama-server --list-devices`, so it works for Metal, CUDA, ROCm,
  Vulkan...), backend, llama.cpp version and commit, threads, and peak bandwidth
  (from a spec table for Apple chips and common GPUs, or `--bandwidth <GB/s>`).
- **Pinned models** (repo, revision, file, exact size, sha256): `base`
  Qwen3.5-4B Q4_0 (everyone runs it), then `small` Gemma 4 E4B, `medium`
  gpt-oss-20b, `large` Qwen3.8-27B UD-IQ4_XS. Each tier runs if file + 1 GiB fits
  the memory budget (GPU memory, or RAM minus room for the OS); override with `--tiers`.
- **Fixed settings**: `llama-bench -p 512 -n 128 -r 3 -o json` (parsed as JSON),
  threads = performance/physical cores; then the 24-question eval against a
  `llama-server` it starts on 127.0.0.1 (`-c 4096`) and always stops,
  at temperature 0 with thinking off (gpt-oss: reasoning effort low).
- **Normalized metrics**: effective GB/s = tg × bytes per token, and
  **% of peak** = effective ÷ peak bandwidth. That's how you compare a laptop with
  a GPU or a cloud box. Plus the eval score per category.
- Output: `results/<machine-id>-<date>.json`. A same-day rerun on that machine updates the file in place.
  `--compare` prints the markdown table across every result file.

```bash
python3 tools/mmbench.py --list                    # what it detected, which tiers it will run
python3 tools/mmbench.py --download --tiers base   # 2.6 GB, ~2 minutes of benchmarking
python3 tools/mmbench.py --bandwidth 90            # e.g. a DDR5-5600 dual-channel PC
python3 tools/mmbench.py --compare
```

Current results (`python3 tools/mmbench.py --compare`):

| model | machine | backend | pp512 t/s | tg128 t/s | effective GB/s | % of peak | eval | math | reasoning | knowledge | instructions | code | extraction | multilingual | eval time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b-q4_0 | apple-m4-pro-20gpu-24gb-metal | Metal | 830.5 | **69.1** | 178 | 65% | **18/24** | 3/4 | 4/6 | 4/4 | 3/4 | 1/3 | 2/2 | 1/1 | 4s |
| qwen3.5-4b-q4_0 | cloud-xeon-4vcpu-15gb-cpu * | CPU | 57.0 | **11.0** | 28 | - | **16/24** | 2/4 | 4/6 | 4/4 | 2/4 | 1/3 | 2/2 | 1/1 | 30s |
| gemma-4-e4b-qat-q4_0 | apple-m4-pro-20gpu-24gb-metal | Metal | 1335.1 | **66.1** | 187 | 68% | **18/24** | 3/4 | 4/6 | 4/4 | 3/4 | 1/3 | 2/2 | 1/1 | 3s |
| gemma-4-e4b-qat-q4_0 | cloud-xeon-4vcpu-15gb-cpu * | CPU | 139.0 | **11.4** | 32 | - | **18/24** | 3/4 | 4/6 | 4/4 | 3/4 | 1/3 | 2/2 | 1/1 | 20s |
| gpt-oss-20b-mxfp4 | apple-m4-pro-20gpu-24gb-metal | Metal | 925.9 | **80.7** | 208 | 76% | **24/24** | 4/4 | 6/6 | 4/4 | 4/4 | 3/3 | 2/2 | 1/1 | 17s |
| gpt-oss-20b-mxfp4 | cloud-xeon-4vcpu-15gb-cpu * | CPU | 37.0 | **10.0** | 26 | - | **24/24** | 4/4 | 6/6 | 4/4 | 4/4 | 3/3 | 2/2 | 1/1 | 157s |
| qwen3.8-27b-iq4_xs | apple-m4-pro-20gpu-24gb-metal | Metal | 125.4 | **13.7** | 185 | 68% | **21/24** | 3/4 | 5/6 | 4/4 | 3/4 | 3/3 | 2/2 | 1/1 | 19s |

\* The cloud VM row is imported from the measurements above (sections 1 and 4),
made before mmbench existed. It uses the same files and llama.cpp commit, but its
eval ran with `labs/03_mini_eval.py` defaults, and no stddev was recorded.

**Validation**: on the M4 Pro, mmbench reproduced the separate `bench.sh` +
`labs/03_mini_eval.py` runs: decode within 1.5%, prefill within the run-to-run
spread, identical scores, and 95 of 96 answers word for word.

**Tested on**: macOS 26 (Metal), and Linux aarch64 in a Debian 13 container with a CPU build
(full base-tier run). **Not yet tested on native Windows.** The Windows code paths
(registry CPU name, `GlobalMemoryStatusEx`, PowerShell core count, `.exe` lookup in
`build/bin/Release`) are written but unverified, so please report what breaks.
