# kt-kernel for the Flash-Next GGUF serve, as two patches

The CPU side of the tree in `../flash-next-gguf/`: kt-kernel's LLAMAFILE
backend with the i-quant kernels, the launch-time and task-queue changes the
serve relies on, and the GGUF types the vendored ggml does not know.

```
git clone https://github.com/kvcache-ai/ktransformers.git && cd ktransformers
git checkout 95009ea
git submodule update --init third_party/llama.cpp third_party/llamafile third_party/pybind11
git apply --whitespace=nowarn <repo>/trees/kt-kernel/kt-kernel-95009ea-flash-next-gguf.patch
( cd third_party/llama.cpp && git apply --whitespace=nowarn <repo>/trees/kt-kernel/third_party-llama.cpp-a94e6ff-types.patch )
```

Build the extension as the serve does (AVX2 host, one CUDA arch):

```
cmake -S kt-kernel -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_AVX2=ON -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DKTRANSFORMERS_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DPYTHON_EXECUTABLE=<venv>/bin/python
cmake --build build --target kt_kernel_ext -j 16
```

and put `build/kt_kernel_ext.cpython-*.so` next to the installed `kt_kernel`
package, whose Python files (`experts_base.py`, `utils/loader.py`,
`utils/llamafile.py`, `utils/amx.py`) are the ones in this patch
(`kt-kernel/python/`).

`kt-kernel-95009ea-flash-next-gguf.patch` (24 files, against upstream commit
`95009ea`, 2026-10):

- `third_party/llamafile/iqk_mul_mat.inc`: the i-quant dot products on AVX2
  (IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS; the x86 port is
  [kvcache-ai/ktransformers#2240](https://github.com/kvcache-ai/ktransformers/pull/2240)),
  the `DequantizerIQ2XS` index fix, and the Q2_0 / IQ4_LUT paths;
  `tinyblas_cpu_sgemm.inc` dispatches them.
- `operators/llamafile/moe.hpp`: the many-phases expert loop (one task
  queue phase per call instead of per expert group), the group-min-len and
  small-block choices for 2-4 token windows, the fused down-merge, THP-backed
  pools; `cpu_backend/`: the task queue and worker pool changes they need,
  `kt_knobs.h` for the live knobs read per call.
- `operators/avx2/`: grouped FP4 fast paths, ModelOpt NVFP4 expert loading,
  Gemma 4 BF16 GeGLU experts (the serve's other models share this tree).
- `python/`: `utils/loader.py` reads a GGUF header without gguf-py
  (`KT_GGUF_FAST_HEADER=1`), resolves a mixed checkpoint's MTP layer, and
  with `KT_GGUF_NOCOPY=1` hands the C++ load views of the file mapping;
  `utils/llamafile.py` the same for the quantized expert tensors;
  `experts_base.py` as upstream at 95009ea plus the newer-Torch allowance.

`third_party-llama.cpp-a94e6ff-types.patch` (against the submodule's pinned
commit `a94e6ff`): ggml type 42 (`Q2_0`, 64 values in 18 bytes, Strata's
draft-expert format) and 43-45 (`IQ4_LUT`, a 4-bit lookup type the CPU
experts use for IQ4_NL / IQ4_XS layers) in `ggml.h` / `ggml-common.h` /
`ggml-quants.{h,c}` / `ggml.c`, and the same ids in `gguf-py` so the loader
can read such files. These stay out of llama.cpp proper (the PR plan moves
them into kt-kernel's own copies).

Both patches are the serving tree's `git diff` against those commits with
backups removed, the Python files taken from the installed package (which
is what runs); applied on a clean worktree they reproduce the tree file for
file (checked 2026-10-13).
