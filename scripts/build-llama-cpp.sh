#!/usr/bin/env bash
# build-llama-cpp.sh - clone (or update) and build llama.cpp for this machine.
#
#   scripts/build-llama-cpp.sh            # auto-detect backend
#   BACKEND=cpu scripts/build-llama-cpp.sh
#   BACKEND=cuda|metal|vulkan|hip|cpu
#
# Result: vendor/llama.cpp/build/bin/{llama-cli,llama-server,llama-bench,...}
#
# Shortcuts if you don't want to compile:
#   macOS:  brew install llama.cpp
#   any OS: prebuilt zips at https://github.com/ggml-org/llama.cpp/releases
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${LLAMA_CPP_DIR:-$ROOT/vendor/llama.cpp}"
JOBS="${JOBS:-$( (nproc || sysctl -n hw.ncpu) 2>/dev/null )}"

for tool in git cmake; do
  command -v "$tool" >/dev/null || { echo "Missing '$tool'. Ubuntu: sudo apt install -y git cmake build-essential  |  macOS: xcode-select --install && brew install cmake"; exit 1; }
done

if [[ -d "$DIR/.git" ]]; then
  echo ">> Updating $DIR"
  git -C "$DIR" pull --ff-only
else
  echo ">> Cloning llama.cpp into $DIR"
  mkdir -p "$(dirname "$DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$DIR"
fi

backend="${BACKEND:-auto}"
if [[ "$backend" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then backend=metal
  elif command -v nvcc >/dev/null || [[ -x /usr/local/cuda/bin/nvcc ]]; then backend=cuda
  elif command -v hipcc >/dev/null; then backend=hip
  elif command -v glslc >/dev/null && command -v vulkaninfo >/dev/null; then backend=vulkan
  else backend=cpu
  fi
fi

flags=(-DCMAKE_BUILD_TYPE=Release)
case "$backend" in
  metal)  ;;  # Metal is on by default on macOS
  cuda)   export PATH="/usr/local/cuda/bin:$PATH"; flags+=(-DGGML_CUDA=ON) ;;
  hip)    flags+=(-DGGML_HIP=ON) ;;
  vulkan) flags+=(-DGGML_VULKAN=ON) ;;
  cpu)    flags+=(-DGGML_NATIVE=ON) ;;  # tune for *this* CPU (AVX2/AVX-512/AMX/NEON)
  *) echo "Unknown BACKEND=$backend"; exit 1 ;;
esac

echo ">> Backend: $backend   (cmake ${flags[*]})"
cmake -S "$DIR" -B "$DIR/build" "${flags[@]}" > "$DIR/build-config.log" 2>&1 \
  || { tail -30 "$DIR/build-config.log"; exit 1; }
echo ">> Compiling with $JOBS jobs (a few minutes the first time)..."
cmake --build "$DIR/build" -j "$JOBS" \
  --target llama-app llama-cli llama-server llama-bench llama-perplexity llama-quantize 2>&1 | grep -E "error|Built target" || true

bin="$DIR/build/bin"
if [[ -x "$bin/llama-server" ]]; then
  echo ">> Done. Binaries in: $bin"
  "$bin/llama-server" --version 2>&1 | tail -2
  echo
  echo "Tip: add it to PATH for this shell:  export PATH=\"$bin:\$PATH\""
else
  echo "!! Build failed - see output above"; exit 1
fi
