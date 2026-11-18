# The Local-LLM Roadmap: from "what's a GGUF?" to running your own models anywhere

Eight phases. Each one has a **goal**, the **ideas** to learn, the **hands-on work**,
and a **"done when"** checklist. Do them in order, because each phase uses the
previous one. Times assume evenings and weekends.

| Phase | You learn to... | Time |
|---|---|---|
| [0. Set up your bench](#phase-0--set-up-your-bench-30-min) | install the tools, know your hardware | 30 min |
| [1. First contact](#phase-1--first-contact-1-2-hours) | run a model and chat with it, three different ways | 1-2 h |
| [2. How it actually works](#phase-2--how-it-actually-works-a-weekend) | predict memory and speed *before* you download | a weekend |
| [3. Choosing models](#phase-3--choosing-models-a-few-evenings) | pick the right model and quant for a task and machine | a few evenings |
| [4. Master the runtime](#phase-4--master-the-runtime-a-week) | tune llama.cpp (and know Ollama, LM Studio and MLX) | a week |
| [5. Build things](#phase-5--build-things-1-2-weeks) | APIs, streaming, JSON, tools, RAG, UIs | 1-2 weeks |
| [6. Cloud GPUs](#phase-6--cloud-gpus-a-weekend) | rent a GPU, serve a model, tunnel to it, shut it down | a weekend |
| [7. Going deeper](#phase-7--going-deeper-ongoing) | quantize and fine-tune your own, serve many users | ongoing |

Reference docs you'll keep coming back to:
[how it works](docs/01-how-it-works.md) ·
[model guide](docs/02-models.md) ·
[tools & cheat sheets](docs/03-tools.md) ·
[cloud playbook](docs/04-cloud.md) ·
[measurements](docs/05-measurements.md)

---

## Phase 0 — Set up your bench (30 min)

**Goal:** a working toolchain and an honest picture of your hardware.

**Do**
1. Clone this repo and run the hardware check:
   ```bash
   git clone <this repo> modelmania && cd modelmania
   scripts/detect-hardware.sh
   ```
   Write down three numbers: **RAM**, **VRAM** (or "unified memory" on a Mac),
   and **memory bandwidth** (look up your chip or GPU in
   [docs/01](docs/01-how-it-works.md#memory-bandwidth-cheat-sheet)). Those three
   numbers predict almost everything that follows.
2. Install llama.cpp. Pick one:
   - `curl -LsSf https://llama.app/install.sh | sh`. This is the official installer and gives you the `llama` command. Read the script first if you like.
   - `brew install llama.cpp` (macOS/Linux), `winget install llama.cpp` (Windows)
   - `scripts/build-llama-cpp.sh` compiles from source and auto-picks CUDA, Metal, ROCm, Vulkan or CPU. It's the best way to learn what the backends are.
3. Python 3.10+ (labs use only the standard library). Optional: `pip install gguf` for `tools/memcalc.py`.

**Done when**
- [ ] `llama-server --version` (or `llama version`) prints a version
- [ ] you know your RAM / VRAM / bandwidth and which **tier** (T0-T5) `detect-hardware.sh` put you in

---

## Phase 1 — First contact (1-2 hours)

**Goal:** talk to a model running entirely on your machine, and see that the
same model can be served by different tools that all speak the same API.

**Do**
1. **The engine directly (llama.cpp).** Use the model for your tier
   ([docs/02](docs/02-models.md#by-hardware-tier)). For a first run, pick something tiny:
   ```bash
   llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF        # downloads, then serves
   # open http://127.0.0.1:9931 : a built-in chat UI
   ```
   Or the repo way, with files you control:
   ```bash
   scripts/get-model.sh unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_0.gguf
   scripts/serve.sh models/Qwen3.5-4B-Q4_0.gguf --no-think
   python3 labs/01_chat.py
   ```
2. **The friendly wrapper (Ollama).** Install from ollama.com, then
   `ollama run gemma4:e4b` (or `qwen3.5:4b`, `gpt-oss:20b`). Note how it hides
   the file, quant and context settings. That's convenient now and confusing later.
3. **The GUI (LM Studio).** Install, search a model, load it, chat. Turn on its
   local server (port 1234).
4. Point the same lab at each one and watch it work unchanged:
   ```bash
   LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=gemma4:e4b python3 labs/01_chat.py   # Ollama
   LLM_BASE_URL=http://127.0.0.1:1234/v1  python3 labs/01_chat.py                       # LM Studio
   ```

**Observe** (lab 01 prints these after every reply)
- *time to first token* gets longer as the conversation grows
- *tokens/second* stays roughly flat
- the model has no memory: the whole history is re-sent on every turn (`/history`)
- with `--think` (or `/think`), the model writes a hidden reasoning section first.
  It's smarter but much slower, which matters a lot on a CPU.

**Done when**
- [ ] you've chatted with a local model in all three tools
- [ ] you can explain why lab 01 works against all three (one OpenAI-compatible API)
- [ ] you know where the model file lives on disk and how big it is

---

## Phase 2 — How it actually works (a weekend)

**Goal:** predict, *before downloading*, whether a model fits and how fast it
will run, and then confirm it by measuring. This is the core skill. Read
[docs/01-how-it-works.md](docs/01-how-it-works.md) alongside.

**Learn**
- tokens and the tokenizer; a model only ever predicts the next token
- the inference loop: **prefill** (read the prompt, compute-bound) vs **decode**
  (write tokens one by one, memory-bandwidth-bound)
- memory = **weights** (≈ file size) + **KV cache** (grows with context) + overhead
- **speed ceiling ≈ memory bandwidth ÷ bytes read per token**
- **quantization**: FP16 → Q8_0 → Q6_K → Q4_K_M/Q4_0 → IQ3/Q2, and what you lose
- **MoE** (mixture of experts): big total size, small *active* size, so it's fast
- **hybrid / sliding-window attention**: why new models have much smaller KV caches
- **thinking models**: more tokens spent means better answers and slower replies

**Do** (each experiment: *predict first, then measure*)
1. Tokens: `python3 labs/02_next_token.py --tokens "Tokenization of 12345 strawberries 🍓"`
2. Next-token probabilities and temperature: `python3 labs/02_next_token.py "The capital of Egypt is"`
3. Memory: `python3 tools/memcalc.py models/<file>.gguf --ctx 4096 32768 131072 --budget <your GiB>`,
   then start `llama-server` with `--log-verbosity 4` and compare the
   `llama_kv_cache: size = ...` line. It should match exactly.
4. Speed: `python3 tools/memcalc.py models/<file>.gguf --bandwidth <GB/s>` predicts
   the ceiling; `scripts/bench.sh models/<file>.gguf` measures it.
5. Prefill vs decode: `PP=4096 scripts/bench.sh models/<file>.gguf`. Prefill
   tokens/s is many times higher than decode tokens/s. Why?
6. Quantization trade-off: download the same model as `Q4_0`, `Q4_K_M` and `Q8_0`.
   Compare size, speed (`bench.sh`) and quality:
   ```bash
   llama-perplexity -m models/<file>.gguf -f wiki.test.raw -c 512 --chunks 16
   ```
   (wiki.test.raw comes from llama.cpp's `scripts/get-wikitext-2.sh`)
7. MoE vs dense: benchmark a ~9B dense model against `gpt-oss-20b`. Which is faster, and why?
8. Thinking: `python3 labs/03_mini_eval.py` then `python3 labs/03_mini_eval.py --think`.
   Compare score and time.

> On this repo's reference machine (4 CPU cores, no GPU) these experiments produced
> some counter-intuitive results: the 21B MoE model was faster *and* smarter than
> the 9B dense one, and `Q4_0` beat `Q4_K_M` on speed with the same quality. See
> [docs/05-measurements.md](docs/05-measurements.md).

**Done when**
- [ ] you can estimate a model's RAM/VRAM need in your head (params × bits ÷ 8 + KV)
- [ ] your speed predictions land within ~2× of the measurement
- [ ] you can explain why a 20B MoE can beat a 9B dense model on speed
- [ ] you can explain what `Q4_K_M` means and when you'd pick `Q8_0` instead

---

## Phase 3 — Choosing models (a few evenings)

**Goal:** go from "what's popular" to "what's right for *my* task and *my*
hardware", backed by your own evidence. Read [docs/02-models.md](docs/02-models.md).

**Learn**
- the families in late 2026: **Qwen3.5 / 3.6 / 3.8**, **Gemma 4**, **gpt-oss**,
  plus Mistral (Devstral, Mistral Small) and Phi. Their sizes, strengths and licenses.
- where weights come from: official repos vs community quantizers (unsloth,
  bartowski, ggml-org), QAT releases, and how to spot junk merges with clickbait names
- reading a model card: architecture, context length, recommended sampling,
  chat template, thinking switch, license
- why leaderboards mislead (contamination, benchmark-tuning, different settings)

**Do**
1. Run `labs/03_mini_eval.py` on 3+ models that fit your tier; `--compare` prints a table.
2. **Write your own eval**: add 15-30 questions from *your* real use-cases to
   `labs/evals/mine.jsonl` (same format) and rerun. This is the most valuable
   file you'll write in this course.
3. Try a vision model: Gemma 4 / Qwen3.5 GGUFs ship an `mmproj-*.gguf` file.
   `llama serve -hf unsloth/gemma-4-E4B-it-GGUF` loads it automatically. Paste an image in the web UI.
4. Read two model cards end-to-end and note their recommended temperature, top_p and top_k values.

**Done when**
- [ ] you have a personal shortlist: a fast model, a smart model, a coding model, and an embedding model
- [ ] your own eval file exists and you've used it to reject at least one hyped model

---

## Phase 4 — Master the runtime (a week)

**Goal:** know every important knob, and what each one costs in memory or speed.
Read [docs/03-tools.md](docs/03-tools.md).

**Learn and try, in llama.cpp**
- `-c` context size and `-np` parallel slots (which share the context!)
- `-ngl` / `--n-cpu-moe` (GPU offload; put MoE experts on the CPU when VRAM is short).
  Since v1, `-ngl auto` and `--fit on` are the defaults: llama.cpp sizes things to
  fit your memory by itself. Learn what it chose (read the log), then override it.
- `-fa` flash attention, `-ctk q8_0 -ctv q8_0` (a quantized KV cache halves its memory)
- `-t` threads (more isn't always faster), `--load-mode mmap|mlock|none`
- `--reasoning-budget`, `--chat-template-kwargs`, `--jinja`
- speculative decoding (`--spec-default`, draft models, `--spec-type ngram-*`)
- router mode: `llama-server --models-dir models/` serves many models and swaps them on demand
- `llama fit-params` computes settings that fit a model in your memory
- sampling: temperature, top_p, top_k, min_p, presence/repeat penalties, and why
  model cards recommend specific values

**Then the alternatives**: map each knob to its Ollama form (Modelfile `PARAMETER`,
`OLLAMA_*` env vars, `ollama ps`), to LM Studio's settings panel, and to MLX on
Apple Silicon (`mlx_lm.server`).

**Do**
1. Find the largest context you can run for your favourite model. Check the
   memcalc prediction against the log, with f16 and q8_0 KV.
2. Benchmark `-t 2/4/8/all` and find your machine's sweet spot.
3. If you have a GPU: load a model that's *slightly* too big with partial `-ngl`,
   and an MoE with `--n-cpu-moe`. Compare the speeds.
4. Serve two models with router mode and switch between them by `model` name.

**Done when**
- [ ] you can explain every flag in your own `serve.sh` line
- [ ] you know the context limit and tokens/s of your favourite model on your machine

---

## Phase 5 — Build things (1-2 weeks)

**Goal:** use local models as a component in real programs.

**Do** (each lab is short. Read the code, it's the point)
| Lab | Skill |
|---|---|
| `labs/llm.py` | the whole OpenAI-compatible protocol in ~100 lines of stdlib Python |
| `labs/01_chat.py` | streaming, history and timing |
| `labs/04_structured.py` | JSON Schema constrained output; valid ≠ true |
| `labs/05_tools.py` | the tool-calling loop that every "agent" is built on |
| `labs/06_rag.py` | embeddings + retrieval + grounded answers over your own docs |

Then build with real-world tools:
- **A chat UI for the family**: Open WebUI (`pip install open-webui`, or Docker)
  pointed at your llama-server, Ollama or vLLM endpoint.
- **A coding assistant**: point an editor or terminal coding tool that supports
  OpenAI-compatible endpoints at your server. Use a coding-capable model from
  [docs/02](docs/02-models.md#by-task), and keep expectations realistic for small models.
- **Speech, vision, embeddings**: Qwen3-ASR / Whisper for speech-to-text, the
  `mmproj` vision models, `--embedding` and `--rerank` servers.

**Done when**
- [ ] you've built one small tool you actually use (a document Q&A, an extractor, a classifier...)
- [ ] you've measured its latency and decided whether it needs a bigger machine (that's Phase 6)

---

## Phase 6 — Cloud GPUs (a weekend)

**Goal:** run bigger models (or serve many users) on a rented GPU, using the same
skills and the same labs, and never get surprised by the bill.
Follow [docs/04-cloud.md](docs/04-cloud.md).

**Learn**
- choosing a GPU: **VRAM decides what fits, bandwidth decides speed**
  (a $0.40/h 24 GB card vs a $2/h 80 GB card)
- providers: marketplace (Vast.ai), pods (RunPod), on-demand clouds (Lambda,
  hyperscalers), serverless (Modal, RunPod Serverless, HF Inference Endpoints)
- **llama.cpp vs vLLM/SGLang**: one user and quantized GGUF, versus many users,
  high throughput, and safetensors / FP8 / AWQ weights
- networking and security: bind to localhost, use an SSH tunnel, require API keys,
  never leave an open port
- cost control: persistent volumes for weights, auto-stop, spot vs on-demand

**Do**
1. Rent a 24 GB GPU (RTX 4090 / L4 class) for an hour. `cloud/bootstrap-gpu.sh`
   sets it up with llama.cpp + CUDA.
2. Tunnel it to your laptop (`ssh -N -L 9931:127.0.0.1:9931 ...`) and run
   `labs/03_mini_eval.py` against it unchanged.
3. Serve the *unquantized* model with vLLM (`cloud/bootstrap-gpu.sh vllm Qwen/Qwen3.5-9B`)
   and load-test it with `vllm bench serve`. See what continuous batching does
   to total throughput.
4. Try one model that's impossible at home (gpt-oss-120b or Qwen3.6-35B-A3B at Q8 on an 80 GB card).
5. Tear everything down and check the billing page.

**Done when**
- [ ] you can go from zero to a tunneled, authenticated endpoint in < 20 minutes
- [ ] you can estimate the cost of a job before you start it
- [ ] you know when cloud beats buying hardware for *your* usage (break-even math in docs/04)

---

## Phase 7 — Going deeper (ongoing)

Pick the threads that interest you:
- **Make your own quants**: `convert_hf_to_gguf.py` → `llama-quantize` (with an
  `--imatrix` from `llama-imatrix`), then measure perplexity and KL-divergence
  against the BF16 original.
- **Fine-tune**: LoRA / QLoRA with Unsloth or TRL on a rented GPU. Merge the
  result, convert it to GGUF and serve it with your own stack. Start with a narrow
  task you can evaluate with your Phase 3 eval file.
- **Throughput engineering**: vLLM/SGLang tensor and pipeline parallelism,
  prefix caching, speculative decoding (MTP/EAGLE), FP8/NVFP4 on Hopper/Blackwell.
- **Production basics**: Prometheus `--metrics`, request logging, rate limits,
  model version pinning, evals in CI.
- **Read the source**: `ggml` kernels, llama.cpp's `llama-graph.cpp`, the vLLM scheduler.

**Capstone ideas**
1. *Private document assistant*: RAG over your notes, served on your machine, with an eval set.
2. *Same app, two homes*: run it locally on a small model, and on a cloud GPU with a big one,
   switching only `LLM_BASE_URL`.
3. *Your own fine-tune*: a small model trained on one narrow task that beats a 4× larger
   general model on your eval.

---

### How to know you've reached "complete familiarity"
You can look at any model release and, within five minutes, say whether it will
run on a given machine, roughly how fast, which file to download, how to serve it,
and how to check that it's actually good for the job, then do all of it either
locally or on a rented GPU.
