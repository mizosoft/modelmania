# Tools: what to use, and cheat sheets

## The landscape in one table

| Tool | What it is | Use it when | Weights |
|---|---|---|---|
| **llama.cpp** (`llama`, `llama-server`, `llama-cli`, `llama-bench`) | *the* local inference engine (v1.0 in 2026). CPU, CUDA, Metal, ROCm, Vulkan, SYCL | you want control, newest features, best CPU/Mac/mixed performance; **this repo's default** | GGUF |
| **Ollama** | a friendly daemon + model library, built on llama.cpp (plus an MLX engine on Apple Silicon) | you want `ollama run X` simplicity, or an app expects Ollama | GGUF (its own registry, or `hf.co/...`) |
| **LM Studio** | desktop GUI (llama.cpp + MLX backends), with an OpenAI-compatible server on :1234 | you want point-and-click, a model browser, per-model settings | GGUF, MLX |
| **mlx-lm** | Apple's MLX framework for LLMs | Apple Silicon, and you want MLX-format models or fine-tuning on a Mac | MLX |
| **vLLM** | high-throughput GPU server (PagedAttention, continuous batching, tensor parallel) | NVIDIA/AMD GPUs serving **many users**, or full-precision/FP8 models | safetensors (BF16/FP8/AWQ/GPTQ…) |
| **SGLang** | high-throughput GPU server, strong on structured/agentic workloads | the same niche as vLLM; try both | safetensors |
| **transformers** (`transformers serve`) | the reference implementation | research, models nobody else supports yet | safetensors |
| **Open WebUI** | a ChatGPT-like web UI for any OpenAI/Ollama endpoint | a nice UI for you or your household/team | n/a |

They're **all OpenAI-API compatible**, so `labs/llm.py` (and any OpenAI SDK)
works against every one of them by changing `LLM_BASE_URL`:

| Server | Default URL |
|---|---|
| llama.cpp (`llama serve` / `llama-server`) | `http://127.0.0.1:9931/v1` (port was 8080 before v1; old tutorials still say 8080) |
| Ollama | `http://127.0.0.1:11434/v1` (needs `LLM_MODEL=<name:tag>`) |
| LM Studio | `http://127.0.0.1:1234/v1` |
| vLLM / SGLang / mlx_lm.server | `http://127.0.0.1:8000/v1` (mlx: `--port` to choose) |

---

## llama.cpp cheat sheet

### Install / update
```bash
curl -LsSf https://llama.app/install.sh | sh    # official installer -> `llama` command; `llama update` later
brew install llama.cpp                          # macOS / Linux
winget install llama.cpp                        # Windows
scripts/build-llama-cpp.sh                      # from source (auto-detects CUDA/Metal/HIP/Vulkan/CPU)
```

### The commands
```bash
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF          # download + serve (web UI at :9931)
llama serve -hf unsloth/Qwen3.5-4B-GGUF:Q4_0        # pick a quant with :TAG
llama cli   -hf unsloth/gemma-4-E4B-it-GGUF         # terminal chat; vision models: type /image photo.jpg
llama download -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M   # just download (to the cache; `llama serve -cl` lists it)
llama bench -m models/x.gguf                        # pp/tg speed
llama fit-params -m models/x.gguf                   # what fits in your memory
llama perplexity -m models/x.gguf -f wiki.test.raw  # quality of a quant
# The older per-tool binaries still exist: llama-server, llama-cli, llama-bench, ...
```

### Flags that matter
| Flag | Meaning | Notes |
|---|---|---|
| `-m FILE` / `-hf repo[:quant]` | model from disk / from Hugging Face | `-hf` also fetches `mmproj` for vision models |
| `-c N` | context tokens to allocate | memory grows ~linearly (see memcalc) |
| `-np N` | parallel request slots | total KV memory is still set by `-c`; the slots share it |
| `-ngl N\|auto\|all` | layers on the GPU | default `auto` + `--fit on` = sized for your VRAM |
| `--n-cpu-moe N` / `--cpu-moe` | keep MoE experts of N layers (or all) in RAM | best trick for big MoE on small GPUs |
| `-fa on` | flash attention | usually `auto` already |
| `-ctk q8_0 -ctv q8_0` | quantized KV cache | ~½ KV memory, tiny quality cost |
| `-t N` | CPU threads | physical cores; more can be slower |
| `--chat-template-kwargs '{"enable_thinking":false}'` | turn thinking off (Qwen3.5, Gemma 4) | per request: `chat_template_kwargs` |
| `--reasoning-budget N` | cap reasoning tokens | `0` = answer immediately |
| `--spec-default`, `-md draft.gguf`, `--spec-type ngram-mod` | speculative decoding | 1.3-2.5× decode on code/repetitive text |
| `--models-dir DIR` (+ `--models-preset ini`) | **router mode**: many models, loaded on demand by `model` name | `--models-max N` limits how many stay loaded |
| `--embedding` / `--rerank` | embedding / reranking server | use an embedding model |
| `--host 0.0.0.0 --api-key KEY` | listen on the network **with auth** | prefer an SSH tunnel; see docs/04 |
| `--metrics` | Prometheus endpoint `/metrics` | |
| `--load-mode mmap\|mlock\|none` | how weights are loaded | `mlock` pins in RAM (no swapping) |
| `--log-verbosity 4` | detailed logs | shows KV cache and buffer sizes |

Every flag also has an env var (`LLAMA_ARG_CTX_SIZE=32768`, ...), which is handy in Docker.

### Endpoints worth knowing
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`,
`/v1/responses` (OpenAI Responses API), `/v1/messages` (Anthropic Messages API),
`/v1/audio/transcriptions`, `/v1/models`, plus native `/completion` (with `n_probs`),
`/tokenize`, `/props`, `/slots`, `/health`, `/metrics`.

### Structured output, tools, grammars
- `response_format: {"type": "json_schema", ...}` → guaranteed-valid JSON (lab 04)
- `tools: [...]` → tool calls with the model's own template (lab 05)
- `grammar: "<GBNF>"` on `/completion` → any format you can write a grammar for

---

## Ollama cheat sheet
```bash
ollama run gemma4:e4b                    # pull + chat;  also: qwen3.5:4b, qwen3.5:9b, gpt-oss:20b
ollama run hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_0   # any GGUF from Hugging Face
ollama ps                                # what's loaded, how much memory, CPU/GPU split, context
ollama list / pull / rm / show <model>   # `show --modelfile` reveals template + params
OLLAMA_CONTEXT_LENGTH=32768 ollama serve # default context for all models
```
A custom model with your own settings (`Modelfile`):
```
FROM ./models/Qwen3.5-9B-Q4_K_M.gguf
PARAMETER num_ctx 32768
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
```
`ollama create my-qwen -f Modelfile && ollama run my-qwen`

Watch out for: the context Ollama actually loaded (`ollama ps`), and tags that
point to a different quant or size than you expect (`ollama show`).

## LM Studio cheat sheet
- Model search, then **pick the exact quant** (it shows whether it fits)
- Developer tab → *Start server* → `http://127.0.0.1:1234/v1`
- CLI: `lms get <model>`, `lms load <model>`, `lms server start`, `lms ps`
- On Macs it offers both GGUF (llama.cpp) and MLX builds. Benchmark both on your machine.

## Apple Silicon: MLX
```bash
pip install mlx-lm
mlx_lm.generate --model mlx-community/Qwen3.5-9B-4bit --prompt "Hello"
mlx_lm.server   --model mlx-community/Qwen3.5-9B-4bit --port 8000   # OpenAI-compatible
mlx_lm.convert  --hf-path Qwen/Qwen3.5-4B -q                        # make your own 4-bit MLX
```
- llama.cpp (Metal) and MLX trade wins depending on the model and chip. Measure both.
- By default the GPU can use only ~⅔-¾ of unified memory. Advanced:
  `sudo sysctl iogpu.wired_limit_mb=<MB>` raises it until reboot. Leave the OS a few GB.

## Front-ends
- **llama.cpp's built-in UI**: just open the server URL. It does chat, images and settings.
- **Open WebUI**: `pip install open-webui && open-webui serve` (Python 3.11-3.12; or Docker), then add a
  connection to your OpenAI-compatible URL. It adds users, RAG over documents, history and more.
- Editors/coding tools: anything with an "OpenAI-compatible" or "custom endpoint"
  setting works. Many coding agents also accept an Anthropic-style endpoint, which
  llama-server provides at `/v1/messages`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `failed to allocate` / OOM / system freezes | weights + KV don't fit | lower `-c`, `-ctk q8_0 -ctv q8_0`, smaller quant, `--n-cpu-moe` |
| answer is **empty**, `finish_reason: length` | thinking used up `max_tokens` | raise `max_tokens`, disable thinking, `--reasoning-budget` |
| `unknown model architecture` | runtime older than the model | update llama.cpp / Ollama / LM Studio |
| gibberish or endless repetition | wrong chat template, bad sampling, context overflow | use the embedded template (`--jinja`), copy the model card's sampling, check `-c` |
| very slow on a GPU machine | layers on the CPU, or a different GPU used | look for `offloaded N/N layers` in the log; `nvidia-smi` while generating |
| slow on a laptop | power-saving mode, thermal throttling, too many threads | plug in, performance mode, try `-t <physical cores>` |
| `couldn't bind HTTP server socket` | port already in use (an old server is still running) | `pkill -f llama-server` or `PORT=9932` |
| WSL2 sees too little RAM | WSL memory cap | set `memory=` in `%UserProfile%\.wslconfig` |
| first answer slow, later fast | model loading / prompt cache warming | normal; `--load-mode mlock` keeps it resident |
