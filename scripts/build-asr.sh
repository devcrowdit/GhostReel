#!/usr/bin/env bash
# Build the ghostreel-asr helper (whisper.cpp). Picks CUDA when the toolkit is available.
#   GHOSTREEL_ASR_GPU=cuda|vulkan|cpu   force a backend
#   PROFILE=release|dev                 (default release)
set -euo pipefail
cd "$(dirname "$0")/.."
want="${GHOSTREEL_ASR_GPU:-auto}"
nvcc="$(command -v nvcc || true)"
[ -z "$nvcc" ] && [ -x /opt/cuda/bin/nvcc ] && nvcc=/opt/cuda/bin/nvcc
[ -z "$nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ] && nvcc=/usr/local/cuda/bin/nvcc
features=""
if [ "$want" = cuda ] || { [ "$want" = auto ] && [ -n "$nvcc" ] && command -v nvidia-smi >/dev/null; }; then
  root="$(dirname "$(dirname "$nvcc")")"
  export CUDA_PATH="${CUDA_PATH:-$root}" CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$root}" CUDACXX="${CUDACXX:-$nvcc}"
  export PATH="$root/bin:$PATH"
  export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86;89}"
  features="--features cuda"
elif [ "$want" = vulkan ]; then
  features="--features vulkan"
fi
# Full-parallel CUDA builds of whisper.cpp exhaust RAM (see plan §11).
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
profile="${PROFILE:-release}"
echo "building ghostreel-asr ($profile) ${features:-cpu}"
cargo build -j 4 --profile "$profile" -p ghostreel-asr $features
