"""Build the source of the multi-token GGUF MoE GEMV (kt_gguf_moe_multi.cu) from
sgl-kernel's csrc/quantization/gguf (moe_vec.cuh, gguf_kernel.cu, the vllm /
llama.cpp b2899 port; Apache-2.0 / MIT).

sgl_kernel.ggml_moe_a8_vec runs one warp per (route, output row): every route
reads its expert's weights again, and the KT wrapper sends the routes whose
expert lives on the CPU to slot 0 with weight 0, so they are computed too. In an
MTP verify window of 3 tokens x top-10 that is 30 expert reads per layer
(nsys 10-09: 41.5 us per call against 14.5 at one token).

The rewritten kernel keeps the grid (rows x routes) and the per-type vec_dot
functions, but:
  - a route whose id is < 0 (expert on the CPU) writes zeros and returns;
  - only the first route of an expert in the window computes: it collects the
    routes with the same id (at most one per token, KT_MAXM = 8 tokens) and runs
    each weight block against all their activations, so the block comes from
    DRAM once and from L1 for the rest;
  - a warp computes KT_RPW rows (4) and a block has KT_MMV_Y warps (4): the
    expert rows are short (10 quant blocks for gate/up and for down), so with
    one row per one-warp block the kernel was block scheduling and load
    latency, not bandwidth;
  - the member count is a template argument (a switch per block), so the
    weight-side half of vec_dot is computed once for all the routes of the
    expert rather than once per route.
Same sums in the same order per route as the original, so the output matches it
bit for bit (tools/test_gguf_moe_multi.py).

Built with -DKT_PER_ROUTE=1 the window is dropped: every route computes its own
rows at any token count (sgl-kernel's work split), with this file's vec_dot
functions (ggml's IQ3_S grid; sgl-kernel's ggml_moe_a8_vec still has the
pre-release one) and zero rows for CPU routes. One route of the per-route build
and the same route as a member of a window run identical sums.

Q2_0 (ggml type 42, which sgl-kernel lacks) is added to the MoE kernel: the
hot experts' Q2_0 downs stay 2-bit on the GPU instead of a 4x Q8_0 repack.

Usage: python kt-gguf-moe-multi-build.py <sgl-kernel dir> <out .cu>
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
g = src / "csrc/quantization/gguf"
out = Path(sys.argv[2])

moe_vec = (g / "moe_vec.cuh").read_text()
kernel_new = r'''// The rows of one warp against NM activations. NM is a compile-time count so the
// member loop has no condition in it: the weight-side work of vec_dot (loads,
// grid lookups, sign and bit unpacking) is the same for every member and the
// compiler computes it once per block instead of once per member.
template <int NM, typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ __forceinline__ void moe_vec_q_rows(
    const block_q_t* __restrict__ x,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* members,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride,
    const int row0) {
  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const block_q8_1* ys[NM];
#pragma unroll
  for (int m = 0; m < NM; ++m) {
    ys[m] = (const block_q8_1*)(((const int*)vy) + (members[m] / topk) * token_stride);
  }
  float tmp[NM][KT_RPW];
#pragma unroll
  for (int m = 0; m < NM; ++m) {
#pragma unroll
    for (int r = 0; r < KT_RPW; ++r) {
      tmp[m][r] = 0.0f;
    }
  }
  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int iby = i * (qk / QK8_1);  // y block index that aligns with ibx
    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int
#pragma unroll
    for (int r = 0; r < KT_RPW; ++r) {
      if (row0 + r < nrows) {
        const int ibx = (row0 + r) * blocks_per_row + i;  // x block index
#pragma unroll
        for (int m = 0; m < NM; ++m) {
          tmp[m][r] += vec_dot_q_cuda(&x[ibx], &ys[m][iby], iqs);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < NM; ++m) {
#pragma unroll
    for (int r = 0; r < KT_RPW; ++r) {
      if (row0 + r < nrows) {
        float t = tmp[m][r];
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
          t += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), t, mask);
        }
        if (threadIdx.x == 0) {
          dst[members[m] * nrows + row0 + r] = t;
        }
      }
    }
  }
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void moe_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* topk_ids,
    const int topk,
    const int ncols,
    const int nrows,
    const int token_stride) {
  // KT_RPW rows per warp, interleaved so their loads overlap
  const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * KT_RPW;
  const int routes = gridDim.z;
  const int p = blockIdx.z;
  const int expert = topk_ids[p];

  if (row0 >= nrows) {
    return;
  }
  // the expert is on the CPU: this route contributes nothing (its weight is 0)
  if (expert < 0) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int r = 0; r < KT_RPW; ++r) {
        if (row0 + r < nrows) {
          dst[p * nrows + row0 + r] = 0;
        }
      }
    }
    return;
  }
  int members[KT_MAXM];
  int nm = 0;
#if KT_PER_ROUTE
  // per-route build (any number of tokens): every route computes its own rows,
  // sgl-kernel ggml_moe_a8_vec's work split; one member runs the same sums in the
  // same order as that member of a window
  (void)routes;
  members[nm++] = p;
#else
  // the first route of this expert computes all of them
  for (int q = 0; q < p; ++q) {
    if (topk_ids[q] == expert) {
      return;
    }
  }
  for (int q = p; q < routes && nm < KT_MAXM; ++q) {
    if (topk_ids[q] == expert) {
      members[nm++] = q;
    }
  }
#endif
#if KT_PTR_TABLE
  // vx is a per-slot pointer table (int64): slots in borrowed memory live anywhere
  const block_q_t* x = (const block_q_t*)(((const long long*)vx)[expert]);
#else
  const block_q_t* x = ((const block_q_t*)vx) + expert * nrows * (ncols / qk);
#endif
#define KT_ROWS(n) moe_vec_q_rows<n, scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>( \
      x, vy, dst, members, topk, ncols, nrows, token_stride, row0)
  switch (nm) {
    case 1: KT_ROWS(1); break;
#if KT_MAXM >= 2
    case 2: KT_ROWS(2); break;
#endif
#if KT_MAXM >= 3
    case 3: KT_ROWS(3); break;
#endif
#if KT_MAXM >= 4
    case 4: KT_ROWS(4); break;
#endif
#if KT_MAXM >= 5
    case 5: KT_ROWS(5); break;
#endif
#if KT_MAXM >= 6
    case 6: KT_ROWS(6); break;
#endif
#if KT_MAXM >= 7
    case 7: KT_ROWS(7); break;
#endif
#if KT_MAXM >= 8
    case 8: KT_ROWS(8); break;
#endif
  }
#undef KT_ROWS
}
'''
a = moe_vec.index("template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>\nstatic __global__ void moe_vec_q(")
b = moe_vec.index("template <typename scalar_t>\nstatic void moe_vec_q4_0_q8_1_cuda(")
moe_vec = moe_vec[:a] + kernel_new + "\n" + moe_vec[b:]
# the launchers' grid: KT_RPW rows per warp
old_grid = "const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;"
assert moe_vec.count(old_grid) > 10, moe_vec.count(old_grid)
moe_vec = moe_vec.replace(old_grid, "const int block_num_y = (nrows + GGML_CUDA_MMV_Y * KT_RPW - 1) / (GGML_CUDA_MMV_Y * KT_RPW);")

gk = (g / "gguf_kernel.cu").read_text()
qa = gk.index("// Q8 gemv\ntemplate <typename scalar_t>\nstatic __global__ void\nquantize_q8_1(")
qb = gk.index("torch::Tensor ggml_dequantize(")
quant = gk[qa:qb]
wa = gk.index("torch::Tensor ggml_moe_a8_vec(")
depth, i = 0, gk.index("{", wa)
while True:
    if gk[i] == "{":
        depth += 1
    elif gk[i] == "}":
        depth -= 1
        if depth == 0:
            break
    i += 1
wrapper = gk[wa:i + 1]
wrapper = wrapper.replace("torch::Tensor ggml_moe_a8_vec(", "torch::Tensor kt_moe_a8_vec_multi(", 1)
# every output row is written by the kernel (zeros for CPU routes)
n = wrapper.count("at::Tensor Y = torch::zeros(")
assert n == 1, n
wrapper = wrapper.replace("at::Tensor Y = torch::zeros(", "at::Tensor Y = torch::empty(")

# rows per block: sgl-kernel launches one warp per block (GGML_CUDA_MMV_Y 1);
# the expert down (320 columns, 10 blocks a row) is then all block scheduling
common = (g / "ggml-common.h").read_text()
assert "#define GGML_CUDA_MMV_Y 1\n" in common

head = r'''// Generated by patches/kt-gguf-moe-multi-build.py from sgl-kernel csrc/quantization/gguf
// (vllm / llama.cpp b2899 port). Multi-token GGUF MoE GEMV for the KT hot experts.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor_sync((mask), (var), (lane_mask), (width))
#define WARP_SIZE 32
#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))
#ifndef KT_RPW
#define KT_RPW 4  // output rows per warp
#endif
#ifndef KT_MAXM
#define KT_MAXM 4  // tokens per window; an expert appears at most once per token
#endif
#ifndef KT_PER_ROUTE
#define KT_PER_ROUTE 0  // 1: no window, one route per row block (moe_a8_vec_routes)
#endif
#ifndef KT_PTR_TABLE
#define KT_PTR_TABLE 0  // 1: W is an int64 table of per-slot weight addresses
#endif

#include "ggml-common.h"
#include "vecdotq.cuh"

// Q2_0 (ggml type 42): 64 values a block, one fp16 scale, 2-bit codes, value =
// code - 1. Byte b of qs holds elements 4b..4b+3 (code s at bits 2s). One
// thread takes 4 bytes (16 elements) against half a q8_1 block. The -1 is
// taken as (sum of code * y) - (sum of y): subtracting 1 per byte inside a
// packed int would borrow across bytes.
#define QK2_0 64
#define QR2_0 4
#define QI2_0 (QK2_0 / (4 * QR2_0))
#define VDR_Q2_0_Q8_1_MMVQ 1
typedef struct {
  half d;
  uint8_t qs[QK2_0 / 4];
} block_q2_0;
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size");

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& iqs) {
  const block_q2_0* bq = (const block_q2_0*)vbq;
  const uint8_t* q = bq->qs + 4 * iqs;  // blocks are 18 bytes: no aligned int loads
  const block_q8_1* y = bq8_1 + (iqs >> 1);
  const int* u = (const int*)y->qs + (iqs & 1) * 4;
  int sumi = 0, sumy = 0;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int c = q[b];
    const int packed = (c & 3) | (((c >> 2) & 3) << 8) | (((c >> 4) & 3) << 16) | (((c >> 6) & 3) << 24);
    sumi = __dp4a(packed, u[b], sumi);
    sumy = __dp4a(0x01010101, u[b], sumy);
  }
  return __half2float(bq->d) * __low2float(y->ds) * (float)(sumi - sumy);
}

'''
# Q2_0: the Q8_0 launcher and switch case, renamed
la = moe_vec.index("template <typename scalar_t>\nstatic void moe_vec_q8_0_q8_1_cuda(")
lb = moe_vec.index("template <typename scalar_t>", la + 10)
q2_launcher = (moe_vec[la:lb].replace("q8_0_q8_1_cuda", "q2_0_q8_1_cuda").replace("QK8_0", "QK2_0")
               .replace("QI8_0", "QI2_0").replace("block_q8_0", "block_q2_0")
               .replace("VDR_Q8_0_Q8_1_MMVQ", "VDR_Q2_0_Q8_1_MMVQ").replace("vec_dot_q8_0_q8_1", "vec_dot_q2_0_q8_1"))
moe_vec = moe_vec[:lb] + q2_launcher + moe_vec[lb:]
ca = wrapper.index("      case 8:\n")
cb = wrapper.index("        break;\n", ca) + len("        break;\n")
wrapper = wrapper[:cb] + wrapper[ca:cb].replace("case 8:", "case 42:").replace("q8_0_q8_1_cuda", "q2_0_q8_1_cuda") + wrapper[cb:]
out.write_text(head + moe_vec + "\n" + quant + "\n" + wrapper + "\n")
print(f"wrote {out} ({len(head + moe_vec + quant + wrapper)} bytes)")
