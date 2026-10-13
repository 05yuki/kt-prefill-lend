#!/usr/bin/env bash
# The Flash-Next GGUF serve from scratch: the two patched trees, a venv with the versions the serve was measured
# with, kt-kernel's extension built for this CPU and GPU. Idempotent: a step whose result exists is skipped.
#
# usage: tools/install.sh <work dir> [--cuda-arch 120] [--python python3.12] [--jobs 16]
#   <work dir>/sglang        sgl-project/sglang at release/v0.5.19 (0bcd822) + trees/flash-next-gguf
#   <work dir>/ktransformers kvcache-ai/ktransformers at 95009ea + trees/kt-kernel (and its ggml submodule patch)
#   <work dir>/venv          Python 3.12, torch 2.13 cu130, flashinfer-python 0.6.18, sgl-kernel 0.4.6.post1,
#                            triton 3.7.1, transformers 5.12.1, sglang from the tree (editable), kt-kernel
# Then: tools/make-config-dir.py, tools/make-draft-dir.py, tools/serve.py (see the README).
# Build requirements: cmake >= 3.25, ninja, a C++17 compiler, CUDA 13.x toolkit (nvcc on PATH or CUDA_HOME),
# git. The kt-kernel build on a glibc 2.41+ host needs the compat header below (NVCC_PREPEND_FLAGS).
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
work="${1:?usage: install.sh <work dir>}"; shift || true
cuda_arch=120; python=python3.12; jobs=16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-arch) cuda_arch="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$work"; work="$(cd "$work" && pwd)"
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")}"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"

# --- SGLang
if [[ ! -d "$work/sglang/.git" ]]; then
  git clone --filter=blob:none https://github.com/sgl-project/sglang.git "$work/sglang"
fi
( cd "$work/sglang"
  if [[ "$(git rev-parse HEAD)" != 0bcd822* ]]; then git fetch -q origin 0bcd822 && git checkout -q 0bcd822; fi
  if ! grep -q "kt_stream_prefill_gguf" python/sglang/srt/layers/moe/kt_ep_wrapper.py 2>/dev/null; then
    git apply --whitespace=nowarn "$repo/trees/flash-next-gguf/sglang-v0.5.19-flash-next-gguf.patch"
    echo "sglang: tree patch applied"
  else echo "sglang: tree already patched"; fi )

# --- ktransformers (kt-kernel)
if [[ ! -d "$work/ktransformers/.git" ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git "$work/ktransformers"
fi
( cd "$work/ktransformers"
  if [[ "$(git rev-parse HEAD)" != 95009ea* ]]; then git fetch -q origin 95009ea && git checkout -q 95009ea; fi
  git submodule update -q --init third_party/llama.cpp third_party/llamafile third_party/pybind11
  if ! grep -q "kt_knobs" kt-kernel/cpu_backend/cpuinfer.h 2>/dev/null; then
    git apply --whitespace=nowarn "$repo/trees/kt-kernel/kt-kernel-95009ea-flash-next-gguf.patch"
    ( cd third_party/llama.cpp && git apply --whitespace=nowarn "$repo/trees/kt-kernel/third_party-llama.cpp-a94e6ff-types.patch" )
    echo "ktransformers: patches applied"
  else echo "ktransformers: already patched"; fi )

# --- venv
if [[ ! -x "$work/venv/bin/python" ]]; then
  "$python" -m venv "$work/venv"
fi
venv="$work/venv"; py="$venv/bin/python"
"$py" -m pip install -q --upgrade pip
"$py" -m pip install -q "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu130
"$py" -m pip install -q "flashinfer-python==0.6.18" "sgl-kernel==0.4.6.post1" "triton==3.7.1" "transformers==5.12.1" \
  "gguf==0.19.0" "safetensors" "ninja" "cmake" "pybind11"
"$py" -m pip install -q -e "$work/sglang/python" --no-deps
"$py" -m pip install -q -r "$work/sglang/python/pyproject.toml" 2>/dev/null || true  # the tree's own deps, best effort
"$py" -m pip install -q -e "$work/ktransformers/kt-kernel" --no-deps --no-build-isolation 2>/dev/null \
  || "$py" -m pip install -q "$work/ktransformers/kt-kernel" --no-deps --no-build-isolation
# the installed kt_kernel Python must be the patched tree's
cp "$work/ktransformers/kt-kernel/python/"*.py "$("$py" -c 'import kt_kernel,os;print(os.path.dirname(kt_kernel.__file__))')/"
cp "$work/ktransformers/kt-kernel/python/utils/"*.py "$("$py" -c 'import kt_kernel,os;print(os.path.dirname(kt_kernel.__file__))')/utils/"

# --- kt-kernel extension
bld="$work/kt-build"
compat="$repo/tools/cuda13-glibc-compat.h"
[[ -f "$compat" ]] && export NVCC_PREPEND_FLAGS="-U_GNU_SOURCE -D_DEFAULT_SOURCE -include $compat"
cmake -S "$work/ktransformers/kt-kernel" -B "$bld" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DKTRANSFORMERS_USE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" -DPYTHON_EXECUTABLE="$py"
cmake --build "$bld" --target kt_kernel_ext -j "$jobs"
so="$(ls "$bld"/kt_kernel_ext.cpython-*.so)"
dst="$("$py" -c 'import kt_kernel,os;print(os.path.dirname(kt_kernel.__file__))')/$(basename "$so")"
cp "$so" "$dst.new" && mv -f "$dst.new" "$dst"
"$py" -c "import kt_kernel.kt_kernel_ext as e; print('kt_kernel_ext', e.__file__)"
echo "done: venv $venv, sglang $work/sglang, ktransformers $work/ktransformers"
echo "next: tools/make-config-dir.py <hf dir> <gguf dir> <config dir>; tools/make-draft-dir.py; tools/serve.py --venv $venv --tree $work/sglang ..."
