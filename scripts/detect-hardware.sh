#!/usr/bin/env bash
# detect-hardware.sh - what can this machine run?
#
# Prints CPU, RAM, GPU/VRAM and a starting recommendation.
# Works on Linux (incl. WSL2) and macOS. Read-only: changes nothing.
set -uo pipefail

os="$(uname -s)"
arch="$(uname -m)"
echo "== System: $os $arch"

ram_gib=0
vram_gib=0
gpu_kind="none"

if [[ "$os" == "Darwin" ]]; then
  cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  cores="$(sysctl -n hw.ncpu)"
  ram_gib=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  echo "== CPU: $cpu ($cores cores)"
  echo "== RAM: ${ram_gib} GiB"
  if [[ "$arch" == "arm64" ]]; then
    gpu_kind="apple"
    # By default macOS lets the GPU use roughly 2/3 (<=36GB) to 3/4 (>36GB) of RAM.
    if (( ram_gib > 36 )); then vram_gib=$(( ram_gib * 3 / 4 )); else vram_gib=$(( ram_gib * 2 / 3 )); fi
    echo "== GPU: Apple Silicon, unified memory (~${vram_gib} GiB usable by the GPU by default)"
  fi
else
  cpu="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  cores="$(nproc)"
  ram_gib=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
  echo "== CPU: $cpu ($cores threads)"
  flags="$(grep -m1 -E '^(flags|Features)' /proc/cpuinfo)"
  simd=""
  for f in avx2 avx512f avx512_vnni avx512_bf16 amx_int8 asimd sve; do
    [[ " $flags " == *" $f "* ]] && simd+="$f "
  done
  echo "   SIMD: ${simd:-none detected}  (llama.cpp uses these; AVX2 is the minimum you want on x86)"
  echo "== RAM: ${ram_gib} GiB"
  grep -qi microsoft /proc/version 2>/dev/null && echo "   (WSL2 detected: RAM shown is what WSL is allowed, see .wslconfig)"

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_kind="nvidia"
    echo "== GPU (NVIDIA):"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
    vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
    vram_gib=$(( vram_mib / 1024 ))
  elif command -v rocm-smi >/dev/null 2>&1; then
    gpu_kind="amd"
    echo "== GPU (AMD/ROCm):"
    rocm-smi --showproductname --showmeminfo vram 2>/dev/null | grep -E "Card series|Total Memory" | sed 's/^/   /'
    vram_b=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
    [[ -n "${vram_b:-}" ]] && vram_gib=$(( vram_b / 1024 / 1024 / 1024 ))
  elif command -v lspci >/dev/null 2>&1 && lspci | grep -qiE 'vga|3d|display'; then
    echo "== GPU: $(lspci | grep -iE 'vga|3d|display' | head -2 | cut -d: -f3- | sed 's/^ //')"
    echo "   (no nvidia-smi / rocm-smi found: drivers missing, or integrated graphics)"
    echo "   Vulkan builds of llama.cpp can still use many integrated/AMD/Intel GPUs."
  else
    echo "== GPU: none detected (CPU-only is fine to start!)"
  fi
fi

disk_free=$(df -h . | awk 'NR==2{print $4}')
echo "== Free disk here: $disk_free (each model is 0.5-20 GB)"

echo
echo "================ Recommendation ================"
# The number that matters: how much fast memory can hold the model.
if [[ "$gpu_kind" == "nvidia" || "$gpu_kind" == "amd" ]]; then
  budget=$vram_gib
  where="VRAM (GPU)"
elif [[ "$gpu_kind" == "apple" ]]; then
  budget=$vram_gib
  where="unified memory"
else
  # leave ~4 GiB for the OS + your apps (or 30% on small machines, whichever is more)
  budget=$(( ram_gib - 4 )); alt=$(( ram_gib * 70 / 100 ))
  (( alt < budget )) || budget=$alt
  where="RAM (CPU-only, after leaving room for the OS)"
fi
echo "Memory budget for model + context: ~${budget} GiB of $where"
echo

if (( budget < 4 )); then
  tier="T0"; pick="Qwen3.5-0.8B / Qwen3.5-2B (Q4_K_M), Gemma 4 E2B"
elif (( budget < 8 )); then
  tier="T1"; pick="Qwen3.5-4B (Q4), Gemma 4 E4B (QAT Q4_0)"
elif (( budget < 14 )); then
  tier="T2"; pick="Qwen3.5-9B (Q4_K_M/Q6_K), Gemma 4 E4B, gpt-oss-20b if >=12 GiB"
elif (( budget < 22 )); then
  tier="T3"; pick="gpt-oss-20b (MXFP4), Gemma 4 12B (QAT), Qwen3.5-9B (Q8_0)"
elif (( budget < 40 )); then
  tier="T4"; pick="Qwen3.8-27B / Qwen3.6-27B (Q4), Gemma 4 26B-A4B or 31B (Q4), Qwen3.6-35B-A3B"
else
  tier="T5"; pick="Qwen3.6-35B-A3B (Q8), Qwen3.8-27B (Q8), gpt-oss-120b (needs ~64 GB)"
fi
echo "Tier $tier -> start with: $pick"
echo "See docs/02-models.md for the full table."
if [[ "$gpu_kind" == "none" ]]; then
  echo
  echo "CPU-only tips: prefer MoE models (few active params) and Q4_0 quants, keep"
  echo "thinking off for chat, and expect speed ~= memory bandwidth / model size."
fi
