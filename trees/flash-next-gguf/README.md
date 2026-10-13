# The Flash-Next GGUF serve, as one patch

`sglang-v0.5.19-flash-next-gguf.patch` turns upstream SGLang at
`release/v0.5.19` (commit `0bcd822`) into the tree this repo's Flash-Next
numbers were measured on: 176 files, every KT-side module included, no
per-feature patcher to run afterwards. It is the tree, not a set of options:
apply it, build, serve.

```
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 0bcd822
git apply --whitespace=nowarn <this repo>/trees/flash-next-gguf/sglang-v0.5.19-flash-next-gguf.patch
```

What is in it, from the bottom up:

- Qwen4Exp (Flash-Next) enablement on v0.5.19: the model, its QSA indexer,
  GDN fused projections, hyper-connection kernels, PLE, the MTP draft model,
  and the KT wrapper for CPU experts (`kt_ep_wrapper.py`).
- the whole model from a GGUF (`KT_GGUF_DENSE=<gguf dir>`, `kt_gguf_dense.py`):
  dense weights, PLE table and token table read from the same GGUF the CPU
  experts use, the converter's reorderings undone, quantized dense layers
  served in their GGUF types (`KT_GGUF_DENSE_QUANT=1`); the GGUF linear
  improvements in `layers/quantization/gguf.py` (merged shards, prefill
  dequant threshold, the Triton column gather for the GDN out_proj).
- the expert streamer, NVFP4 edition (`kt_stream_prefill.py`) and GGUF edition
  (`kt_stream_prefill_gguf.py`, `kt_gguf_dequant.py`): above
  `KT_GPU_STREAM_PREFILL` tokens the CPU experts are DMA'd from pinned host
  banks and computed on the GPU, with the prefill lend (`kt_lend.py`) giving
  them the decode weights' VRAM.
- the hot-expert cache (`kt_hot_cache.py`, `kt_gguf_hot.py`, the window kernel
  in `kt_gguf_moe_multi/`), the KV-pool page lend (`kt_hot_kv_lend.py`), the
  CPU handoff through UVA with the two NUMA parts summed on the GPU
  (`kt_uva_handoff.py`), the int8 / fp8 hyper-connection mix for decode
  (`kt_hc_mix_v2.py`), the MTP draft's experts from a GGUF (`kt_mtp_gguf.py`,
  `kt_draft_hot.py`, the Q6_K lm_head shared with the draft).
- the low-latency all-reduce through host memory for TP2 decode
  (`ll_allreduce.py`, `host_staged_allreduce.py`), the launch-time pieces
  (`kt_pinned.py`, `kt_gguf_prefetch.py`, `kt_launch_timing.py`), the live
  knobs (`kt_knobs.py`), the memory snapshot hook (`kt_mem_snapshot.py`) — all
  off unless their environment variable is set.
- the QSA Triton kernels cast an fp8 KV cache before their dots
  (`--kv-cache-dtype fp8_e4m3` then works on the extend path; a no-op for bf16).

Not in it: the per-pass speculative-decoding dump (a measurement patch; the
rest of `eagle_worker_v2.py`'s changes, the QSA / MTP wiring, is in), and the
launcher — `tools/serve.py` is the launcher. The knobs the launcher sets are listed in the top-level README
(`Knobs`) and in each module's docstring; `tools/plan-launch.py` prints the
hot-expert count, chunk and thread counts for your card, model and context.

It also needs:

- kt-kernel built from the `kt-kernel` directory of
  [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers)
  with its LLAMAFILE CPU backend, plus the i-quant kernels for AVX2 hosts
  ([kvcache-ai/ktransformers#2240](https://github.com/kvcache-ai/ktransformers/pull/2240),
  until merged) — Strata's IQ3_S Flash-Next GGUF has IQ2_S / IQ3_XXS /
  IQ3_S / IQ4_NL / IQ4_XS experts. `../kt-kernel/` has the CPU side as two
  patches on upstream kt-kernel and its vendored ggml, the same way.
- the model: the GGUF, a config directory for it (`config.json` with
  `quantization_config.quant_method: gguf` and the tokenizer files), and for
  MTP a draft directory (Strata's MTP GGUF laid out as a one-layer draft
  model; the conversion script is not published yet, nor the data).
- the SGLang runtime the tree expects (torch, flashinfer with sm_120 kernels,
  Triton) — the `requirements` of v0.5.19.

The patch is the serving tree's `git diff 0bcd822` with the measurement patch
and backup files removed; it reproduces that tree file for file (checked on
a clean worktree, 2026-10-13), and `tools/install.sh` + `tools/serve.py` on a
fresh clone served the same decode speed as the serving tree (TP1, 84-91
tok/s on the 1,024-token Japanese prompts, pool 65,536). Where a knob's default differs from
upstream, the module docstring says so.
