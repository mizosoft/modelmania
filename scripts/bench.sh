#!/usr/bin/env bash
# bench.sh - measure prompt-processing and generation speed with llama-bench
#
#   scripts/bench.sh                         # every .gguf in models/
#   scripts/bench.sh models/a.gguf models/b.gguf
#   THREADS=8 PP=1024 TG=256 scripts/bench.sh models/a.gguf
#   EXTRA="-ngl 99 -fa 1" scripts/bench.sh   # pass extra llama-bench flags (GPU offload etc.)
#
# Appends a markdown table to results/<machine>/bench-<date>.md
#   (<machine> = $MACHINE_ID if set, e.g. the id `tools/mmbench.py --list` prints; else the hostname)
#   pp = prompt processing (prefill) tokens/s  -> how fast it READS your prompt
#   tg = text generation (decode) tokens/s     -> how fast it WRITES the answer
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bench="$(command -v llama-bench || echo "$ROOT/vendor/llama.cpp/build/bin/llama-bench")"
[[ -x "$bench" ]] || { echo "llama-bench not found: run scripts/build-llama-cpp.sh"; exit 1; }

if (( $# )); then models=("$@"); else
  shopt -s nullglob; models=("$ROOT"/models/*.gguf)
  # skip vision projectors and embedding models
  filtered=()
  for m in "${models[@]}"; do
    [[ "$(basename "$m")" =~ [Mm]mproj|[Ee]mbed ]] || filtered+=("$m")
  done
  models=(${filtered[@]+"${filtered[@]}"})
fi
(( ${#models[@]} )) || { echo "No models found. Try scripts/get-model.sh"; exit 1; }

threads="${THREADS:-$( (nproc || sysctl -n hw.perflevel0.physicalcpu) 2>/dev/null )}"
mkdir -p "$ROOT/results"
machine="${MACHINE_ID:-$(hostname -s)}"
mkdir -p "$ROOT/results/$machine"
out="$ROOT/results/$machine/bench-$(date +%Y%m%d).md"

margs=()
for m in "${models[@]}"; do margs+=(-m "$m"); done

{
  echo
  echo "### $(date '+%Y-%m-%d %H:%M') on $machine, threads=$threads ${EXTRA:-}"
  echo
  # shellcheck disable=SC2086
  "$bench" "${margs[@]}" -t "$threads" -p "${PP:-512}" -n "${TG:-128}" -r "${REPS:-2}" ${EXTRA:-} 2>/dev/null | grep -E '^\|'
} | tee -a "$out"
echo
echo ">> appended to $out"
