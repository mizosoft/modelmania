#!/usr/bin/env bash
# get-model.sh - download one GGUF file from Hugging Face into models/
#
#   scripts/get-model.sh <hf-repo> <file.gguf>
#   scripts/get-model.sh unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
#   scripts/get-model.sh --list unsloth/Qwen3.5-4B-GGUF      # show available files + sizes
#
# Resumes interrupted downloads. Set HF_TOKEN for gated/private repos.
# (llama-server can also download by itself:  llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MODELS_DIR:-$ROOT/models}"
auth=()
[[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

if [[ "${1:-}" == "--list" ]]; then
  repo="${2:?usage: get-model.sh --list <hf-repo>}"
  curl -sSL ${auth[@]+"${auth[@]}"} "https://huggingface.co/api/models/$repo/tree/main" | python3 -c '
import sys, json
for f in json.load(sys.stdin):
    if f["path"].endswith(".gguf"):
        print("  {:6.2f} GB  {}".format(f.get("size", 0) / 1e9, f["path"]))'
  exit 0
fi

repo="${1:?usage: get-model.sh <hf-repo> <file.gguf>   (or --list <hf-repo>)}"
file="${2:?usage: get-model.sh <hf-repo> <file.gguf>}"
mkdir -p "$DEST"
out="$DEST/$(basename "$file")"

echo ">> $repo / $file  ->  $out"
if ! curl -L --fail --progress-bar -C - ${auth[@]+"${auth[@]}"} -o "$out" \
  "https://huggingface.co/$repo/resolve/main/$file"; then
  [[ -s "$out" ]] || rm -f "$out"   # keep partial downloads (re-run resumes), drop empty ones
  echo "!! Download failed. Check the names with: $0 --list $repo"
  echo "   (401/403 = gated model: accept its license on huggingface.co and set HF_TOKEN)"
  exit 1
fi

# Sanity check: every GGUF file starts with the magic bytes "GGUF"
if [[ "$(head -c 4 "$out")" != "GGUF" ]]; then
  echo "!! $out is not a GGUF file (gated repo? wrong name?)"; exit 1
fi
echo ">> OK: $(du -h "$out" | cut -f1)"
