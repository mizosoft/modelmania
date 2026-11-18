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

bw_gbs=""     # peak memory bandwidth (GB/s) from the spec table below, if known
bw_note=""

if [[ "$os" == "Darwin" ]]; then
  cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  cores="$(sysctl -n hw.ncpu)"
  pcores="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
  ecores="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || true)"
  ram_gib=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  if [[ -n "$pcores" && -n "$ecores" ]]; then
    echo "== CPU: $cpu ($cores cores: $pcores performance + $ecores efficiency)"
  else
    echo "== CPU: $cpu ($cores cores)"
  fi
  echo "== RAM: ${ram_gib} GiB (unified: CPU and GPU share it)"
  if [[ "$arch" == "arm64" ]]; then
    gpu_kind="apple"
    gpu_cores="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2; exit}')"
    # The GPU may only wire part of RAM. Default (iogpu.wired_limit_mb=0): about
    # 2/3 of RAM up to 36 GB, 3/4 above. llama.cpp prints the exact figure at load
    # time as "recommendedMaxWorkingSetSize". You can raise it until reboot with
    #   sudo sysctl iogpu.wired_limit_mb=<MiB>
    wired_mb="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
    server="$(command -v llama-server || echo "$(cd "$(dirname "$0")/.." && pwd)/vendor/llama.cpp/build/bin/llama-server")"
    mtl_mib=""
    [[ -x "$server" ]] && mtl_mib="$("$server" --list-devices 2>&1 | sed -nE 's/^ *MTL0: .*\(([0-9]+) MiB,.*/\1/p' | head -1)"
    if [[ -n "$mtl_mib" ]]; then
      vram_gib=$(( mtl_mib / 1024 )); vram_src="$mtl_mib MiB, measured by llama.cpp (Metal recommendedMaxWorkingSetSize)"
    elif (( wired_mb > 0 )); then
      vram_gib=$(( wired_mb / 1024 )); vram_src="iogpu.wired_limit_mb=$wired_mb (set by you)"
    else
      # Rough estimate. Measured: 74% on a 24 GB M4 Pro (macOS 26); older macOS used ~2/3.
      if (( ram_gib > 36 )); then vram_gib=$(( ram_gib * 3 / 4 )); else vram_gib=$(( ram_gib * 2 / 3 )); fi
      vram_src="estimate; build llama.cpp for the exact figure"
    fi
    vram_src+="; raise with: sudo sysctl iogpu.wired_limit_mb=<MiB>"
    echo "== GPU: Apple Silicon${gpu_cores:+, $gpu_cores GPU cores}, Metal backend"
    echo "   usable by the GPU: ~${vram_gib} GiB of ${ram_gib} GiB ($vram_src)"
    # Peak bandwidth, Apple spec sheets. Binned Max chips have less (fewer memory channels).
    case "$cpu" in
      "Apple M1")       bw_gbs=68 ;;   "Apple M1 Pro")  bw_gbs=200 ;;
      "Apple M1 Max")   bw_gbs=400 ;;  "Apple M1 Ultra") bw_gbs=800 ;;
      "Apple M2")       bw_gbs=100 ;;  "Apple M2 Pro")  bw_gbs=200 ;;
      "Apple M2 Max")   bw_gbs=400 ;;  "Apple M2 Ultra") bw_gbs=800 ;;
      "Apple M3")       bw_gbs=100 ;;  "Apple M3 Pro")  bw_gbs=150 ;;
      "Apple M3 Max")   if [[ "${gpu_cores:-0}" -ge 40 ]]; then bw_gbs=400; else bw_gbs=300; fi ;;
      "Apple M3 Ultra") bw_gbs=819 ;;
      "Apple M4")       bw_gbs=120 ;;  "Apple M4 Pro")  bw_gbs=273 ;;
      "Apple M4 Max")   if [[ "${gpu_cores:-0}" -ge 40 ]]; then bw_gbs=546; else bw_gbs=410; fi ;;
      "Apple M5")       bw_gbs=153 ;;
    esac
    [[ -n "$bw_gbs" ]] && bw_note="Apple spec sheet for $cpu${gpu_cores:+ ($gpu_cores-core GPU)}"
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
    gname="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    case "$gname" in   # peak GB/s, NVIDIA spec sheets (same as docs/01)
      *"RTX 5090"*) bw_gbs=1792 ;; *"RTX 4090"*) bw_gbs=1008 ;; *"RTX 3090"*) bw_gbs=936 ;;
      *"RTX 4070"*) bw_gbs=504 ;;  *"4060 Ti"*) bw_gbs=288 ;;  *"RTX 3060"*) bw_gbs=360 ;;
      *L40S*) bw_gbs=864 ;; *L4*) bw_gbs=300 ;; *A100*80GB*) bw_gbs=2039 ;;
      *H100*) bw_gbs=3350 ;; *H200*) bw_gbs=4800 ;;
    esac
    [[ -n "$bw_gbs" ]] && bw_note="spec sheet for $gname"
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
if [[ -n "$bw_gbs" ]]; then
  echo "== Peak memory bandwidth: ${bw_gbs} GB/s ($bw_note; real decode reaches ~60-80%)"
else
  echo "== Peak memory bandwidth: unknown. Look up your chip/RAM in docs/01 (bandwidth cheat sheet)"
fi
py_ok="$(python3 -c 'import sys; print(int(sys.version_info >= (3, 10)))' 2>/dev/null || echo 0)"
if [[ "$py_ok" != 1 ]]; then
  echo "== Python: '$(command -v python3 || echo none)' is $(python3 -V 2>&1 | cut -d' ' -f2); labs want 3.10+"
  for p in python3.14 python3.13 python3.12 python3.11 python3.10; do
    command -v "$p" >/dev/null && { echo "   found $(command -v "$p"): use it, e.g. $p -m venv .venv"; break; }
  done
fi

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
