#!/usr/bin/env bash
# serve.sh - start llama-server (OpenAI-compatible API + web chat UI)
#
#   scripts/serve.sh models/Qwen3.5-4B-Q4_0.gguf
#   scripts/serve.sh models/Qwen3.5-9B-Q4_K_M.gguf -c 32768 --no-think
#   PORT=9932 scripts/serve.sh models/embeddinggemma-2-Q8_0.gguf --embedding
#
# Then open http://127.0.0.1:9931 (chat UI) or point any OpenAI client at
# http://127.0.0.1:9931/v1   (api key: anything, unless you pass --api-key)
#
# --no-think  : convenience flag -> turns off "thinking" for Qwen3.5/Gemma 4 style
#               templates. Huge speed-up for chat on slow hardware.
# Everything else is passed straight to llama-server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
model="${1:?usage: serve.sh <model.gguf> [llama-server args...]}"
shift

server="$(command -v llama-server || true)"
[[ -z "$server" && -x "$ROOT/vendor/llama.cpp/build/bin/llama-server" ]] && server="$ROOT/vendor/llama.cpp/build/bin/llama-server"
[[ -z "$server" ]] && { echo "llama-server not found: run scripts/build-llama-cpp.sh (or brew install llama.cpp)"; exit 1; }

extra=()
for a in "$@"; do
  if [[ "$a" == "--no-think" ]]; then
    extra+=(--chat-template-kwargs '{"enable_thinking":false}')
  elif [[ "$a" == "--embedding" || "$a" == "--embeddings" ]]; then
    # embedding models must see each input in ONE micro-batch: raise it from 512
    extra+=("$a" -b 2048 -ub 2048)
  else
    extra+=("$a")
  fi
done

# Defaults you can override by passing the same flag again (last one wins):
#   -c 16384 : context window (tokens). Bigger = more memory (see tools/memcalc.py)
#   -np 1    : one conversation slot -> all context goes to you
#   --host 127.0.0.1 : only this machine can connect. Do NOT expose 0.0.0.0
#                      to the internet without --api-key (and ideally a tunnel).
exec "$server" -m "$model" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-9931}" \
  -c 16384 -np 1 \
  ${extra[@]+"${extra[@]}"}
