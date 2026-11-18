# Cloud GPU playbook

Everything you learned locally carries over. A rented GPU is just "a machine
with more memory bandwidth" that you reach over SSH. The labs don't change: you
only point `LLM_BASE_URL` somewhere else.

## When to go to the cloud

- the model doesn't fit at home (27B+ at good quality, 120B-class MoE)
- you need speed (long documents, prefill-heavy RAG, coding agents)
- you serve **several users at once** (vLLM's batching turns one GPU into many)
- you're **fine-tuning** (needs a GPU and much more memory than inference)
- you want to try hardware before buying it

When **not** to: one person chatting with a model that already runs at home.
A rented GPU generating 50 tok/s for one user costs ~$3 per million tokens,
which is often more than a hosted API. Rent for throughput, privacy or control,
not for convenience.

## Step 1: pick the GPU (VRAM decides what fits, bandwidth decides speed)

Use `tools/memcalc.py` with the GPU's VRAM as `--budget` and its bandwidth from
[docs/01](01-how-it-works.md#memory-bandwidth-cheat-sheet).

| You want to run | Minimum VRAM | Typical GPUs | Ballpark $/hour (Nov 2026)* |
|---|---|---|---|
| ≤ 9B models, learning, a GGUF Q4 27B | 24 GB | RTX 4090 / 3090, L4, A10G | $0.25 - 0.90 |
| 27-35B at Q4-Q6, gpt-oss-20b for many users | 32-48 GB | RTX 5090 (32 GB), L40S / RTX 6000 Ada (48 GB) | $0.40 - 1.10 |
| 27-35B at BF16/FP8, gpt-oss-120b, 70B Q4 | 80-96 GB | A100 80 GB, H100 80 GB, RTX PRO 6000 (96 GB) | $1.20 - 4.00 |
| 120B+ dense or big MoE at high precision, max throughput | 141-192 GB, or several GPUs | H200, B200, MI300X; 2-8× H100 | $3 - 6 per GPU |

\*Snapshot from public price trackers in Oct/Nov 2026. Marketplace prices vary
daily and by host. Treat these as orders of magnitude and check the provider's
page before you start.

Rules of thumb:
- L4 is cheap but slow (300 GB/s). An RTX 4090 at a similar price is ~3× faster at decode.
- For a single user, a 4090/5090 is the price/performance sweet spot. For many users, H100-class wins on $/token.
- MoE models (gpt-oss-120b, Qwen3.6-35B-A3B) are the cheapest big-model experience: they
  need the VRAM, but they generate quickly.

## Step 2: pick the provider type

| Type | Examples | Good for | Watch out for |
|---|---|---|---|
| Marketplace | Vast.ai | cheapest hours, wide GPU choice | variable reliability; hosts are third parties |
| GPU pods | RunPod (Community / Secure), TensorDock | quick start with CUDA templates and persistent volumes | community tiers have no SLA |
| GPU clouds | Lambda, CoreWeave, Nebius, Crusoe | reliable on-demand H100/B200, clusters | higher price, sometimes waitlists |
| Hyperscalers | AWS (g6/g6e/p5), GCP (g2/a3), Azure (NC/ND) | org compliance, integration | quotas must be requested; pricier on-demand |
| Serverless / managed | Modal, RunPod Serverless, HF Inference Endpoints, HF Jobs | scale-to-zero APIs, batch jobs, no SSH | cold starts; less control |

Start with a pod or marketplace box with SSH: it's the most educational, and it
matches this repo's scripts.

## Step 3: the 20-minute workflow

```bash
# 0. once: an SSH key (add the .pub to the provider's settings)
ssh-keygen -t ed25519 -C cloud-gpu

# 1. launch: Ubuntu + CUDA (or a "PyTorch" template), 1 GPU, 80-150 GB disk.
#    Attach a persistent volume (often /workspace) so weights survive restarts.

# 2. connect and check the GPU
ssh -p <port> root@<ip>
nvidia-smi

# 3. set up and serve (binds to 127.0.0.1 only, with an API key)
git clone <this repo> modelmania && cd modelmania
cloud/bootstrap-gpu.sh llamacpp unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL           # GGUF route
# or
cloud/bootstrap-gpu.sh vllm Qwen/Qwen3.5-9B --reasoning-parser qwen3        # vLLM route

# 4. on your laptop: tunnel, then use it like a local server
ssh -N -L 9931:127.0.0.1:9931 -p <port> root@<ip>
export LLM_BASE_URL=http://127.0.0.1:9931/v1 LLM_API_KEY=<printed key> LLM_MODEL=<model>
python3 labs/03_mini_eval.py --label cloud-qwen3.8-27b

# 5. DONE? stop (or delete) the instance. Check the billing page.
```

`cloud/bootstrap-gpu.sh` was tested end-to-end on CPU (`ALLOW_CPU=1`) for the
llama.cpp route. The CUDA build and the vLLM route follow the official install
steps but weren't run on a GPU while this repo was written, so read the log if
something fails.

### Without the script (to understand what it does)
```bash
# llama.cpp with CUDA, in Docker
docker run --gpus all -p 127.0.0.1:9931:9931 -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/<file>.gguf --host 0.0.0.0 --port 9931 -c 32768 -np 4 --api-key "$KEY"

# vLLM
uv venv --python 3.12 && source .venv/bin/activate && uv pip install vllm
vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 --api-key "$KEY" \
  --max-model-len 32768 --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

## llama.cpp vs vLLM/SGLang on a GPU

| | llama.cpp | vLLM / SGLang |
|---|---|---|
| Weights | GGUF (heavily quantized OK) | safetensors BF16/FP8/AWQ/GPTQ/NVFP4 (GGUF support is limited) |
| Best at | 1-4 users, fitting big models in small VRAM, CPU+GPU split | many concurrent users, max tokens/s per GPU, multi-GPU tensor parallel |
| Memory | allocates what `-c`/`-np` need | **pre-allocates ~90% of VRAM** (`--gpu-memory-utilization`) for the KV cache. That's expected. |
| Startup | seconds | minutes (weight loading + CUDA graph capture) |
| Benchmark with | `llama-bench`, `labs/03_mini_eval.py` | `vllm bench serve`; watch total throughput as concurrency rises |

Useful vLLM flags: `--max-model-len`, `--tensor-parallel-size N` (N GPUs),
`--quantization fp8`, `--enable-prefix-caching`, `--reasoning-parser`,
`--enable-auto-tool-choice --tool-call-parser <family>`, `--speculative-config`.
Model cards (Qwen, Gemma) list exact vLLM/SGLang commands. Copy those.

## Security checklist (read before you start)

- [ ] servers bind to **127.0.0.1**; you connect through an **SSH tunnel**
- [ ] if you *must* expose a port: `--api-key`, the provider's firewall, ideally HTTPS via a reverse proxy
- [ ] SSH: keys only, no passwords
- [ ] `HF_TOKEN` read-only scope, set as an env var, never baked into images or committed
- [ ] remember the provider's host can see your prompts. Use "secure"/certified tiers for sensitive data.
- [ ] community/marketplace machines are someone else's computer: no long-lived secrets on them

## Cost control

- **Persistent volume + `HF_HOME` on it** means you download 50 GB of weights once, not every session
- set the provider's **auto-stop / idle shutdown**. Put a calendar reminder on anything you leave running.
- spot/interruptible instances are 30-70% cheaper; fine for experiments, bad for demos
- *storage bills even when the GPU is stopped*. Delete volumes you no longer need.

**Break-even math (rent vs buy):** a ~$2,000 24-32 GB card vs renting at ~$0.50/h
gives ~4,000 hours. At 2 h/day that's 5+ years, so rent. At 8 h/day, or for a
24/7 service, buy (and remember power costs). Also compare against hosted APIs
for the same open models, which are often cheaper per token than a single-user rental.

## Fine-tuning in the cloud (Phase 7 preview)

- QLoRA on a 24 GB GPU handles up to ~14B; LoRA on an 80 GB GPU handles 27-35B
- tools: **Unsloth** (fast, memory-efficient, notebooks), **TRL** (Hugging Face), Axolotl
- loop: dataset → train LoRA → evaluate on *your* eval set → merge → `convert_hf_to_gguf.py`
  → `llama-quantize` → serve it with the same `scripts/serve.sh` at home
