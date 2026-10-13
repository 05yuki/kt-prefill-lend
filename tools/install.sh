#!/usr/bin/env bash
# The Flash-Next GGUF serve from scratch: the two patched trees, a venv with the versions the serve was measured
# with, kt-kernel's extension built for this CPU and GPU. Idempotent: a step whose result exists is skipped.
#
# usage: tools/install.sh <work dir> [--cuda-arch 120] [--python python3.12] [--jobs 16]
#   <work dir>/sglang        sgl-project/sglang at release/v0.5.19 (0bcd822) + trees/flash-next-gguf
#   <work dir>/ktransformers kvcache-ai/ktransformers at 95009ea + trees/kt-kernel (and its ggml submodule patch)
#   <work dir>/venv          Python 3.12, torch 2.13 cu130, flashinfer-python 0.6.18, sglang-kernel 0.4.6.post1, torch_memory_saver 0.0.9.post1,
#                            triton 3.7.1, transformers 5.12.1, sglang from the tree (editable), kt-kernel
# Then: tools/make-config-dir.py, tools/make-draft-dir.py, tools/serve.py (see the README).
# Build requirements: a C++17 compiler, CUDA 13.x toolkit (nvcc on PATH or CUDA_HOME), git, and the dev packages
# kt-kernel links against: libnuma-dev libhwloc-dev (apt) — or --sysroot DIR with them unpacked under DIR/usr
# (dpkg -x of the .deb files) when apt is not yours to run. cmake and ninja come from the venv. The kt-kernel build
# on a glibc 2.41+ host needs the compat header beside this script (NVCC_PREPEND_FLAGS).
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
work="${1:?usage: install.sh <work dir>}"; shift || true
cuda_arch=120; python=python3.12; jobs=16; sysroot=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sysroot) sysroot="$2"; shift 2 ;;
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
sglang_base=0bcd822377da7b5718e674eaf9c870d349424dd1   # release/v0.5.19
kt_base=95009ea6856c0799e517e93cb12be5e8494bc7ce       # ktransformers main, 2026-10
( cd "$work/sglang"
  if [[ "$(git rev-parse HEAD)" != "$sglang_base" ]]; then
    git fetch -q origin "$sglang_base" || git fetch -q origin
    git checkout -q "$sglang_base" || { echo "sglang: cannot check out $sglang_base" >&2; exit 3; }
  fi
  if [[ ! -f python/sglang/srt/layers/moe/kt_stream_prefill_gguf.py ]]; then
    git apply --whitespace=nowarn "$repo/trees/flash-next-gguf/sglang-v0.5.19-flash-next-gguf.patch" \
      || { echo "sglang: the tree patch did not apply (is HEAD $sglang_base?)" >&2; exit 3; }
    echo "sglang: tree patch applied"
  else echo "sglang: tree already patched"; fi )

# --- ktransformers (kt-kernel)
if [[ ! -d "$work/ktransformers/.git" ]]; then
  git clone --filter=blob:none https://github.com/kvcache-ai/ktransformers.git "$work/ktransformers"
fi
( cd "$work/ktransformers"
  if [[ "$(git rev-parse HEAD)" != "$kt_base" ]]; then
    git fetch -q origin "$kt_base" || git fetch -q origin
    git checkout -q "$kt_base" || { echo "ktransformers: cannot check out $kt_base" >&2; exit 3; }
  fi
  git submodule update -q --init third_party/llama.cpp third_party/llamafile third_party/pybind11
  if [[ ! -f kt-kernel/cpu_backend/kt_knobs.h ]]; then
    git apply --whitespace=nowarn "$repo/trees/kt-kernel/kt-kernel-95009ea-flash-next-gguf.patch"
    ( cd third_party/llama.cpp && git apply --whitespace=nowarn "$repo/trees/kt-kernel/third_party-llama.cpp-a94e6ff-types.patch" )
    echo "ktransformers: patches applied"
  else echo "ktransformers: already patched"; fi )

# --- venv (Python 3.12: on PATH, from uv, or an explicit --python)
if ! command -v "$python" >/dev/null 2>&1; then
  for c in "$(command -v uv >/dev/null 2>&1 && uv python find 3.12 2>/dev/null)" \
           "$HOME"/.local/share/uv/python/cpython-3.12*/bin/python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12; do
    [[ -n "$c" && -x "$c" ]] && { python="$c"; break; }
  done
fi
command -v "$python" >/dev/null 2>&1 || { echo "no Python 3.12 found; pass --python <path>" >&2; exit 3; }
echo "python: $("$python" --version) at $(command -v "$python")"
if [[ ! -x "$work/venv/bin/python" ]]; then
  "$python" -m venv "$work/venv"
fi
venv="$work/venv"; py="$venv/bin/python"
"$py" -m pip install -q --upgrade pip
"$py" -m pip install -q "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu130
"$py" -m pip install -q "flashinfer-python==0.6.18" "sglang-kernel==0.4.6.post1" "torch_memory_saver==0.0.9.post1" "triton==3.7.1" "transformers==5.12.1" \
  "gguf==0.19.0" "safetensors" "ninja" "cmake" "pybind11"
# sglang from the tree with its runtime dependencies, the pinned packages held by a constraints file (so pip does
# not replace torch); the Rust extensions are not used here
printf '%s\n' "torch==2.13.0" "torchvision==0.28.0" "flashinfer-python==0.6.18" "sglang-kernel==0.4.6.post1" \
  "triton==3.7.1" "transformers==5.12.1" "torch_memory_saver==0.0.9.post1" > "$work/constraints.txt"
SGLANG_BUILD_RUST_EXTS=none "$py" -m pip install -q -e "$work/sglang/python" -c "$work/constraints.txt"
# the kt_kernel Python package: the patched tree's python/ copied as site-packages/kt_kernel (pip would rebuild
# the whole extension with its own cmake; the extension is built below instead)
site="$("$py" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
rm -rf "$site/kt_kernel"
cp -r "$work/ktransformers/kt-kernel/python" "$site/kt_kernel"
find "$site/kt_kernel" -name "*.bak-*" -delete
cat > "$site/kt_kernel-0.7.0.dist-info-marker" <<EOF2
kt_kernel copied from $work/ktransformers/kt-kernel/python by tools/install.sh
EOF2

# --- kt-kernel extension (the venv's cmake / ninja on PATH)
export PATH="$venv/bin:$PATH"
bld="$work/kt-build"
compat="$repo/tools/cuda13-glibc-compat.h"
[[ -f "$compat" ]] && export NVCC_PREPEND_FLAGS="-U_GNU_SOURCE -D_DEFAULT_SOURCE -include $compat"
sysroot_args=()
if [[ -n "$sysroot" ]]; then
  sysroot_args=(-DCMAKE_PREFIX_PATH="$sysroot/usr"
    -DCMAKE_CXX_FLAGS="-I$sysroot/usr/include -I$sysroot/usr/include/x86_64-linux-gnu"
    -DCMAKE_LIBRARY_PATH="$sysroot/usr/lib/x86_64-linux-gnu")
  export PKG_CONFIG_PATH="$sysroot/usr/lib/x86_64-linux-gnu/pkgconfig:$sysroot/usr/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
fi
cmake -S "$work/ktransformers/kt-kernel" -B "$bld" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DKTRANSFORMERS_USE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" -DPYTHON_EXECUTABLE="$py" \
  "${sysroot_args[@]}"
cmake --build "$bld" --target kt_kernel_ext -j "$jobs"
so=""
shopt -s nullglob
for f in "$bld"/kt_kernel_ext.cpython-*.so "$bld"/_kt_kernel_ext_*.so; do so="$f"; break; done
shopt -u nullglob
[[ -n "$so" ]] || { echo "no kt_kernel_ext .so in $bld" >&2; exit 3; }
dst="$site/kt_kernel/$(basename "$so")"
cp "$so" "$dst.new" && mv -f "$dst.new" "$dst"
"$py" -c "import kt_kernel.kt_kernel_ext as e; print('kt_kernel_ext', e.__file__)"
echo "done: venv $venv, sglang $work/sglang, ktransformers $work/ktransformers"
echo "next: tools/make-config-dir.py <hf dir> <gguf dir> <config dir>; tools/make-draft-dir.py; tools/serve.py --venv $venv --tree $work/sglang ..."
