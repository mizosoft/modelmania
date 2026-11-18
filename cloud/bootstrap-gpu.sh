#!/usr/bin/env bash
# bootstrap-gpu.sh - turn a fresh rented GPU box (Ubuntu + NVIDIA driver) into a
# private, authenticated LLM endpoint you reach through an SSH tunnel.
#
#   cloud/bootstrap-gpu.sh llamacpp unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL   # GGUF via llama.cpp (CUDA)
#   cloud/bootstrap-gpu.sh llamacpp models/my-model.gguf                    # a local GGUF file
#   cloud/bootstrap-gpu.sh vllm Qwen/Qwen3.5-9B --reasoning-parser qwen3    # safetensors via vLLM
#
# Environment (all optional):
#   PORT      9931 (llamacpp) / 8000 (vllm)
#   CTX       context length, default 32768
#   NP        parallel slots for llama.cpp, default 4
#   API_KEY   generated if unset; saved to ~/.llm-api-key
#   HF_HOME   model cache; defaults to /workspace/hf when /workspace exists (persistent volume)
#   HF_TOKEN  for gated models
#   ALLOW_CPU=1  run the llamacpp path without a GPU (for testing the script itself)
#
# The server binds to 127.0.0.1 ONLY. Reach it from your laptop with:
#   ssh -N -L <PORT>:127.0.0.1:<PORT> <user>@<host> [-p <ssh-port>]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
engine="${1:?usage: bootstrap-gpu.sh llamacpp|vllm <model> [extra server args]}"
model="${2:?usage: bootstrap-gpu.sh llamacpp|vllm <model> [extra server args]}"
shift 2

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
CTX="${CTX:-32768}"
[[ -d /workspace && -z "${HF_HOME:-}" ]] && export HF_HOME=/workspace/hf
LOGDIR="${HF_HOME:-$HOME}/logs"; mkdir -p "$LOGDIR"

# ---- 1. GPU check ----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
  backend=cuda
elif [[ "${ALLOW_CPU:-0}" == "1" && "$engine" == "llamacpp" ]]; then
  echo ">> No GPU, ALLOW_CPU=1: testing on CPU"; backend=cpu
else
  echo "!! nvidia-smi not found. Use a GPU image with the NVIDIA driver (e.g. a CUDA / PyTorch template)."
  exit 1
fi

# ---- 2. API key -------------------------------------------------------------
if [[ -z "${API_KEY:-}" ]]; then
  if [[ -f ~/.llm-api-key ]]; then API_KEY="$(cat ~/.llm-api-key)"
  else API_KEY="sk-local-$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24)"
       (umask 077; echo "$API_KEY" > ~/.llm-api-key)
  fi
fi

wait_healthy() {  # url, seconds
  local url="$1" limit="$2" i
  for ((i = 0; i < limit; i += 5)); do
    curl -sf -H "Authorization: Bearer $API_KEY" "$url" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

# ---- 3. Engine --------------------------------------------------------------
case "$engine" in
  llamacpp)
    PORT="${PORT:-9931}"
    if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null; then
      echo ">> Installing build tools"
      $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git cmake build-essential libssl-dev curl >/dev/null
    fi
    BACKEND=$backend "$ROOT/scripts/build-llama-cpp.sh"
    server="$ROOT/vendor/llama.cpp/build/bin/llama-server"
    if [[ -f "$model" ]]; then src=(-m "$model"); else src=(-hf "$model"); fi
    log="$LOGDIR/llama-server.log"
    echo ">> Starting llama-server (log: $log)"
    nohup "$server" "${src[@]}" --host 127.0.0.1 --port "$PORT" \
      -c "$CTX" -np "${NP:-4}" --api-key "$API_KEY" --metrics "$@" > "$log" 2>&1 &
    health="http://127.0.0.1:$PORT/health"
    model_name="$(basename "$model")"
    ;;
  vllm)
    PORT="${PORT:-8000}"
    venv="${HF_HOME:-$HOME}/venv-vllm"
    if [[ ! -x "$venv/bin/vllm" ]]; then
      echo ">> Installing vLLM into $venv (a few minutes)"
      command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
      uv venv --python 3.12 "$venv"
      VIRTUAL_ENV="$venv" uv pip install -q vllm
    fi
    log="$LOGDIR/vllm.log"
    echo ">> Starting vLLM (log: $log). First start downloads weights + compiles; can take 5-15 min."
    nohup "$venv/bin/vllm" serve "$model" --host 127.0.0.1 --port "$PORT" \
      --api-key "$API_KEY" --max-model-len "$CTX" --gpu-memory-utilization 0.90 "$@" > "$log" 2>&1 &
    health="http://127.0.0.1:$PORT/health"
    model_name="$model"
    ;;
  *) echo "engine must be llamacpp or vllm"; exit 1 ;;
esac

echo ">> Waiting for the server to become healthy..."
if ! wait_healthy "$health" 1800; then
  echo "!! Server did not come up. Last log lines:"; tail -30 "$log"; exit 1
fi

cat <<EOF

================================================================
 Ready: $engine serving $model_name on 127.0.0.1:$PORT (this machine only)
 API key: $API_KEY      (saved in ~/.llm-api-key)

 On YOUR laptop, open a tunnel (keep it running):
   ssh -N -L $PORT:127.0.0.1:$PORT $(whoami)@<this-host-ip> [-p <ssh-port>]

 Then, from your modelmania checkout:
   export LLM_BASE_URL=http://127.0.0.1:$PORT/v1 LLM_API_KEY=$API_KEY LLM_MODEL=$model_name
   python3 labs/01_chat.py
   python3 labs/03_mini_eval.py

 Logs:  tail -f $log
 Stop:  pkill -f '${engine/llamacpp/llama-server}'   ... and STOP THE INSTANCE when done (it bills by the hour)
================================================================
EOF
