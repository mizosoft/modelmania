# How local inference actually works

The goal of this page: after reading it you can look at a model and predict
**whether it fits** and **how fast it will run** on a machine, then check that
prediction by measuring it. Everything here is checked with real numbers in
[05-measurements.md](05-measurements.md).

---

## 1. What a "model" is, physically

A model is **one big file of numbers (weights)** plus a small amount of metadata:
architecture, tokenizer and chat template. A "4B" model has ~4 billion weights.
Running it ("inference") means doing matrix multiplications with those weights,
over and over, using a runtime such as llama.cpp, vLLM or MLX.

File formats you'll meet:
| Format | Used by | Notes |
|---|---|---|
| **GGUF** | llama.cpp, Ollama, LM Studio | single file, weights pre-quantized, metadata + chat template inside |
| **safetensors** | vLLM, SGLang, transformers, MLX | the original Hugging Face format, usually BF16; FP8/AWQ/GPTQ variants for GPUs |
| **MLX** | Apple MLX / mlx-lm | safetensors laid out for Apple Silicon, often 4/8-bit |

"Open-weight" means you can download the weights. It does *not* always mean
open-source training data or code, and **licenses differ** (Apache-2.0, Gemma
terms, custom community licenses). Read the license before you build a product.

---

## 2. Tokens: the only thing a model sees

Text is split into **tokens** (pieces of words) by a tokenizer. One token is
roughly ¾ of an English word, but numbers, code and non-English text often use
more tokens per word.

```
"Tokenization is surprisingly weird: 12345 strawberries!"
-> Token | ization | is | surprisingly | weird | : | ␣ | 1 | 2 | 3 | 4 | 5 | strawberries | !
```
(real output from `labs/02_next_token.py --tokens`. Note the digits are split one by one.)

This explains classic failures: "how many r's in strawberry?" is hard because the
model never sees letters, only `straw`+`berry`-sized chunks.

**Context window** = the maximum number of tokens (prompt + answer + history) the
model can attend to at once. The *trained* limit (e.g. 262,144 for Qwen3.5) and
the limit *you allocate* (`-c 16384`) are different things. You pay memory for
whatever you allocate.

---

## 3. The inference loop

An LLM does one thing: **given the tokens so far, it outputs a probability for
every possible next token.** A sampler picks one, appends it, and the loop
repeats. Chat, reasoning and tool calls are all built from this.

```
The capital of Egypt is ->  ' Cairo' 67.7% | ' located' 3.4% | ' a' 2.5% | ...
```
(real output from `labs/02_next_token.py`)

Two phases with very different performance:

| Phase | What happens | Bottleneck | Metric |
|---|---|---|---|
| **Prefill** (prompt processing, "pp") | all prompt tokens go through the model *in parallel* | **compute** (FLOPs) | prompt tokens/s, time-to-first-token |
| **Decode** (text generation, "tg") | one token at a time; *every weight is read once per token* | **memory bandwidth** | generated tokens/s |

That's why on any machine prefill is many times faster than decode (e.g. 139
vs 11 tokens/s on the reference CPU), and why a long pasted document makes you
wait before the first word appears.

### The speed rule of thumb

```
max decode tokens/s  ≈  memory bandwidth (GB/s)  ÷  bytes read per token (GB)
```

For a dense model, the bytes read per token are roughly the model file size. Examples:
- 5.7 GB model on an M4 Pro (273 GB/s) → ceiling ≈ 48 tok/s
- the same model on a dual-channel DDR5 laptop (~70 GB/s usable) → ≈ 12 tok/s
- the same model on an RTX 4090 (1008 GB/s) → ≈ 175 tok/s

Real numbers land at 50-90% of the ceiling. On machines with **few CPU cores**
you can be compute-bound before you're bandwidth-bound. The reference 4-core
VM shows this: the dequantization work in `Q4_K_M` made it slower than `Q4_0`.

### Sampling

The "temperature" divides the model's scores before picking a token. Low T means
predictable output, and T=0 always takes the top token. High T is more varied and
riskier. `top_p`, `top_k` and `min_p` cut off unlikely tokens. Model cards give
recommended values. For example, Qwen3.5's non-thinking mode uses
`temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`. Use them: bad
sampling settings make good models look bad.

---

## 4. Memory: will it fit?

```
memory  ≈  weights  +  KV cache  +  overhead (compute buffers, ~0.2-1 GB)
```

### Weights ≈ parameters × bits-per-weight ÷ 8

| Quant | ~bits/weight | 8B model | Quality vs original |
|---|---|---|---|
| BF16 / F16 | 16 | 16 GB | reference |
| Q8_0 | 8.5 | 8.5 GB | practically lossless |
| Q6_K | 6.6 | 6.6 GB | near-lossless |
| Q5_K_M | 5.7 | 5.7 GB | very good |
| **Q4_K_M** | ~4.8-5.2 | ~5 GB | **the usual sweet spot** |
| Q4_0 / IQ4_XS / MXFP4 | ~4.3-4.5 | ~4.5 GB | close to Q4_K_M; Q4_0 has very fast CPU/ARM kernels |
| Q3_K_M / IQ3 | ~3.5-3.9 | ~3.7 GB | noticeable loss |
| Q2_K / IQ2 | ~2.5-3 | ~2.8 GB | large loss; only for huge models that won't fit otherwise |

- **K-quants** (`Q4_K_M`...) quantize in blocks and keep sensitive tensors at higher precision.
- **I-quants** (`IQ4_XS`...) and **imatrix** quants use calibration data to decide what to keep.
- **QAT** (quantization-aware training, e.g. Google's `gemma-4-*-qat-q4_0-gguf`) means
  the model was *trained* to survive 4-bit, so it loses less than a post-hoc quant.
- **MXFP4** is the 4-bit float format gpt-oss was released in, and native on new NVIDIA GPUs.
- Rule: **a bigger model at Q4 usually beats a smaller model at Q8.** Below Q4, quality drops quickly.

### KV cache: the part that grows with context

For every token in context, each attention layer stores a Key and a Value vector:

```
KV bytes per token = 2 × (layers with full attention) × (KV heads) × (head dim) × (bytes per value)
```

Classic transformer (Llama-3-8B style: 32 layers × 8 KV heads × 128 dim, f16):
`2 × 32 × 8 × 128 × 2 = 128 KiB/token`, so 32K tokens = **4 GiB** and 128K = **16 GiB**.
At long context the cache can exceed the weights!

Modern models fight this with three tricks. `tools/memcalc.py` knows all three:
- **GQA** (grouped-query attention): few KV heads shared by many query heads.
- **Sliding-window attention (SWA)**: most layers only look at the last N tokens
  (Gemma 4: 512, gpt-oss: 128), so their cache stops growing.
- **Hybrid linear attention** (Qwen3.5/3.6/3.8): only 1 in 4 layers is full attention;
  the rest keep a small fixed-size state. Qwen3.5-9B needs **32 KiB/token**, a
  quarter of the classic layout. Measured: 131K context = exactly 4.00 GiB of KV.

Knobs: `-c` (allocate less context), `-ctk q8_0 -ctv q8_0` (halves the KV),
`-np` (parallel slots *share* `-c`).

### MoE: mixture of experts

An MoE layer has many "expert" sub-networks and a router that picks a few per token.
- gpt-oss-20b: 21B weights in total, **4 of 32 experts per token**, ~3.6B active
- Qwen3.6-35B-A3B: 35B in total, ~3B active ("A3B")
- Gemma 4 26B-A4B: 26B in total, ~4B active

**Memory is set by the total** (all experts must be loaded). **Speed is set by the
active parameters** (only those are read per token). So MoE is the best friend
of CPUs and Macs, which have lots of memory but modest bandwidth. Measured on the
reference VM: gpt-oss-20b (11.3 GB) generated *faster* than the dense Qwen3.5-9B (5.3 GB).

With a GPU that's too small, llama.cpp can keep the attention layers on the GPU
and push the experts to the CPU (`--n-cpu-moe N`). That's often the best way to
run a big MoE on a 12-24 GB card.

### "Effective" parameters (Gemma 4 E2B/E4B)

Gemma's "E" models have large **per-layer embedding** tables that are only
*looked up*, not multiplied, for each token. The E4B file is 4.8 GB (7.5B weights
in total), but it runs like a ~4B model. Measured: 11.4 tok/s, the same as a
2.4 GB Qwen3.5-4B at Q4_0.

---

## 5. Chat templates and thinking

Chat models were trained on conversations wrapped in special tokens. The **chat
template** (stored in the GGUF) turns `[{"role": "user", ...}]` into that exact
text. A wrong template gives you a confused model. llama.cpp applies the embedded
one by default (`--jinja`).

**Thinking / reasoning models** write a hidden reasoning section before the
answer (returned as `reasoning_content`). It's more accurate on hard problems, but
it can cost 10-100× more tokens. On slow hardware, that's the difference between
1 second and 1 minute. Measured on the reference VM, Qwen3.5-4B spent 400+ tokens
"thinking" about a two-sentence question and never got to the answer.

How to control it:
- Qwen3.5 / Gemma 4: `chat_template_kwargs: {"enable_thinking": false}` per request,
  or `--chat-template-kwargs '{"enable_thinking":false}'` on the server
  (`scripts/serve.sh --no-think`)
- gpt-oss: always reasons; set `reasoning_effort: low | medium | high`
- any model: `--reasoning-budget N` caps the thinking tokens

---

## 6. Where your hardware fits

| Hardware | Memory for the model | Bandwidth | What that means |
|---|---|---|---|
| CPU only | system RAM (lots, cheap) | low (40-120 GB/s) | small dense models or MoE; prefill is slow |
| NVIDIA/AMD GPU | VRAM (8-32 GB consumer) | very high (300-1800 GB/s) | fast, but the model must fit in VRAM (or be split) |
| Apple Silicon | unified memory (8-512 GB) | medium-high (120-819 GB/s) | big models fit; good speed; prefill slower than an NVIDIA GPU |
| AMD Strix Halo / NVIDIA DGX Spark | 128 GB unified | ~256-273 GB/s | built for big MoE models at moderate speed |
| Cloud GPU | 24-192 GB per GPU | 300-8000 GB/s | anything, for as long as you pay |

### Memory bandwidth cheat sheet

Approximate peak specs. Real-world use is about 60-90% of these.

| Device | GB/s | | Device | GB/s |
|---|---|---|---|---|
| DDR4-3200 dual channel | 51 | | RTX 3060 12 GB | 360 |
| DDR5-5600 dual channel | 90 | | RTX 4060 Ti 16 GB | 288 |
| Apple M1 / M2 / M3 | 68 / 100 / 100 | | RTX 4070 12 GB | 504 |
| Apple M4 / M5 | 120 / 153 | | RTX 3090 / 4090 (24 GB) | 936 / 1008 |
| Apple M4 Pro | 273 | | RTX 5090 (32 GB) | 1792 |
| Apple M4 Max | 410-546 | | L4 (24 GB) | 300 |
| Apple M3 Ultra | 819 | | L40S (48 GB) | 864 |
| Ryzen AI Max+ 395 (128 GB) | 256 | | A100 80 GB | 2039 |
| DGX Spark / GB10 (128 GB) | 273 | | H100 SXM 80 GB / H200 141 GB | 3350 / 4800 |

The reference VM for this repo (4 vCPUs of a Xeon server) achieved an *effective*
~20-30 GB/s, which is laptop-class.

---

## 7. Glossary

- **Active parameters**: weights used per token (MoE); sets the speed.
- **AWQ / GPTQ / FP8 / NVFP4**: GPU-oriented quantization formats used by vLLM/SGLang.
- **BF16**: 16-bit floating point; the usual original format of released weights.
- **Continuous batching**: a server that mixes many users' requests into one batch (vLLM). Total throughput rises a lot.
- **Embedding model**: turns text into a vector for search/RAG; it doesn't generate text.
- **Flash attention**: a faster, memory-efficient attention kernel (`-fa`).
- **imatrix**: an importance matrix from calibration text, used for better low-bit quants.
- **mmproj**: the vision (or audio) encoder file that makes a GGUF model multimodal.
- **MTP / speculative decoding**: guess several tokens cheaply, verify them in one pass. It speeds up decode.
- **Offloading (`-ngl`)**: how many layers live on the GPU; the rest run on the CPU.
- **Perplexity (PPL)**: how "surprised" a model is by real text. Lower is better. It's useful for comparing quants *of the same model*, not across models.
- **RoPE / YaRN**: positional encoding, and a method to stretch the context beyond the trained length.
- **Tensor parallel (TP)**: one model split across several GPUs (vLLM `--tensor-parallel-size`).
