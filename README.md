# kt-prefill-lend

Lend the GPU-resident weights of a KTransformers + SGLang server to its long
prefills, and take them back before decode.

In a KTransformers hybrid (CPU experts, GPU attention and hot experts), a long
prompt is prefilled by streaming the CPU-resident experts over PCIe into the
GPU, chunk by chunk. Every forward pays a fixed transfer cost, so a larger
chunk is faster, but on a 16 GB card the chunk is capped by what the decode
footprint (attention weights, hot experts, KV pool, CUDA graphs) leaves free.

During such a prefill this patch pauses the decoder-layer weights with
[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) (physical
memory released, virtual addresses kept, so the decode CUDA graphs stay
valid), streams each layer's weights back from a pinned host mirror into one
of two one-layer slots just before that layer runs (the next layer prefetched
on a side stream), and restores everything before the next decode step.
Buffers only a prefill uses (the expert streamer's slots, graph inputs and
workspace, the full-GPU prefill slot) live in a second tagged region that is
resident only during prefill. The prefill gets several GB more, the chunk
grows, and the forward count drops.

The launcher picks the chunk from what the server measured: each prefill
window records free memory, peak growth and rows per rank, and the first
launch of a setup runs one calibration launch (a ~60K-token prompt), halving
the guess on out-of-memory.

## Who this is for

People who already run a KTransformers hybrid (kt-kernel built for their
CPU, experts on the CPU, SGLang serving attention and hot experts on the GPU)
and are comfortable patching an SGLang tree. This is not a packaged install:
each model here runs on its own SGLang tree, the expert streamer the patches
edit is not in upstream SGLang (it is under `streamers/`), and the DeepSeek V4
zero-copy streamer also needs kt-kernel built with shared-memory expert arenas.
Once applied, the lend needs no tuning: it is on by default in the launchers,
the chunk is calibrated on the first launch, and a tree without the patch
starts without it.

## Results

2x RTX 5070 Ti 16 GB (PCIe 4.0 x16, no P2P), dual EPYC 7452, TP=2, all CPU
experts in kt-kernel (AVX2). Prefill in tokens/s; "chunk" is the prefill
chunk; base = the same server without the lend.

| Model | base chunk | lend chunk (auto) | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16, hot 48 | 2048 | 32768 | 1,568 -> **6,492-6,507** | 1,533 -> **6,406** | 91-97 -> 98-100 | fingerprints bit-identical |
| MiMo-V2.6-Flash MXFP4, hot 12 | 2048 | 22528 | 525 -> **1,962-1,978** | 496 -> **1,550** | 30-32 -> 32 | top-20 logprob spread unchanged |
| DeepSeek V4.1-Flash MXFP4, 1M ctx | 2048 | 8192 | 255 -> **636-639** | 259 -> **594** | 25.4-25.6 -> 27.4-27.9 | paired NLL z -1.90 |
| DeepSeek V4-Flash-Vision MXFP4 | 2048, hot 10 | 16384, hot 12 | 601 -> **900-919** | 598 -> **849** | 36.5-39.9 -> 42.5-44.8 | paired NLL z -0.58 (hot 10, chunk 8192) |
| GLM-5.3-Flash NVFP4 experts | 2048, hot 0 | 14336, hot 10 | 163 -> **326-327** | 155 -> **292** | 18.3 -> 19.9-20.1 | paired NLL z -0.35 / -1.04 |
| Qwen3.8-Flash-Next NVFP4, hot 32 | (developed with it) | 16384 | 4,123-4,145 | 4,604-4,608 (105K) | 57.8-59.4 | fingerprints bit-identical to no-lend |

What made the difference per model is how much of a forward is fixed cost:
Qwen3.5 and MiMo forwards are dominated by it (4x), V4.1 and GLM about half
(2x), Vision-Exp's forward already scales with its rows at chunk 8192
(1.4x). Decode never got slower.

GLM-5.3 is the case the lend was not designed for and helps most in kind:
its full-GPU prefill writes every expert of a layer into a 2.11 GiB slot,
which used to be reserved out of the KV pool, so hot experts and the GPU
prefill excluded each other. With the slot in the lend scratch both run in
one server, and the KV pool still grew (198K -> 213K tokens).

Paired NLL: the same token ids read teacher-forced on both servers, per-token
log-prob differences, z of the mean (negative = lend side lower NLL).
"Fingerprints": top-20 next-token logprobs at 312/548/1,512-token prompts on
models that repeat bit for bit.

## Contents

```
kt_lend.py                          the mechanism (model-agnostic)
patches/
  sglang-kt-prefill-lend.py         Qwen3.8-Flash-Next tree (SGLang v0519 fork)
  sglang-dsv41-kt-prefill-lend.py   DeepSeek V4 tree and copies (V4.1, V4-Flash, Vision, MiMo)
  sglang-qwen35-kt-prefill-lend.py  Qwen3.5 tree (bf16 streamer)
  sglang-glm53-kt-prefill-lend.py   GLM-5.3 tree (no streamer; the full-GPU prefill)
  sglang-kt-stream-shared-graph-inputs.py   streamer graph inputs shared across chunk sizes (apply before the lend on the MiMo tree)
  kt-kernel-shared-gpu-output.py    kt-kernel: one GPU output base shared across batch sizes
  kt-kernel-lend-gpu-output.py      kt-kernel: that output in the lend scratch during a prefill
tools/kt-lend-auto.sh               chunk hint / first-launch calibration for a launcher
streamers/                          the expert streamers the patches edit, as they were before the lend
```

Each patch script takes the SGLang tree as its argument, copies `kt_lend.py`
into `layers/moe/`, and edits by exact anchors (it refuses when one is not
found exactly once, and skips a file that already carries its marker).

## How it is wired

Five points, none in a model file:

- `model_loader._initialize_model`: `kt_lend.bind_model(model)` right after the
  model is built. The largest `ModuleList` named `layers` is taken as the
  decoder; per layer every CUDA parameter of at least 1 MiB moves into one
  span allocated in the `kt_lend_weights` tag, layer by layer.
- the weight-processing loop of `load_weights_and_postprocess`:
  `kt_lend.after_module(module)` puts a layer back into its span as soon as a
  repack (Marlin, swizzled scales) has moved its parameters out, so originals
  and repacked copies do not pile up over the load.
- the end of `load_weights_and_postprocess`: `kt_lend.after_load()` lays every
  layer out again from what the load left (fused owners such as Flash-Next's
  `_fused_in_proj_weight` taken in whole, renamed or reshaped parameters
  taken in, aliases counted once), mirrors the spans into pinned memory and
  pauses the scratch, before the KV pool is sized.
- `ModelRunner.forward`, before `_forward_raw`: `kt_lend.before_forward(batch)`
  switches phase. A forward of at least the streamer's threshold (or GLM's
  `--kt-gpu-prefill-token-threshold`) enters prefill; the next shorter one
  returns to decode, after checking the weights fit back (it raises with the
  numbers instead of letting the remap abort).
- the prefill-only buffers: `kt_lend.scratch_empty(...)` / `scratch_region()`.

A layer is entered through its forward pre-hook, a pre-hook on each child
holding lent parameters, or any other `forward*` method of the layer (DeepSeek
V4 calls `forward_hc_pre_from_prev`); entering layer i points layer i-1 back
at its span.

## Requirements and limits

- torch_memory_saver with hook mode `torch` (tested 0.0.9.post1).
- The expert streamer (`streamers/`), or GLM's full-GPU prefill. Without a
  streamed prefill nothing is lent.
- Prefill must run eagerly (no prefill CUDA graphs); decode graphs are fine.
- Tested only with TP=2 on one host. Not tested: EP, DP attention, PP, a
  speculative draft model (a draft built after the target stays resident),
  SGLang's own `--enable-memory-saver`.
- A model whose layers read another layer's weights, or code outside the
  decoder that reads decoder weights during prefill, faults (illegal address)
  instead of reading stale data. `KT_PREFILL_LEND_SKIP` (default `engram`)
  keeps matching parameters resident.
- The chunk is fixed at launch; the hint applies to the next launch. A window
  that never filled a chunk may only lower it, and a launch raises it at most
  twofold.
- On models whose context pool grows with the chunk (DeepSeek V4's SWA pool)
  a larger chunk costs some full-attention KV (V4.1: 1,031,680 tokens instead
  of 1M at chunk 8192).

Knobs: `KT_PREFILL_LEND=1`, `KT_PREFILL_LEND_MIN_BYTES` (1 MiB),
`KT_PREFILL_LEND_SKIP`, `KT_PREFILL_LEND_IDLE` (lent but not brought back
during prefill; measured within 1% on GLM), `KT_PREFILL_LEND_HINT` (hint file
prefix), `KT_PREFILL_LEND_MARGIN_MB` (512), `KT_PREFILL_LEND_DEBUG=1`
(per-window memory log and a torch memory snapshot when a window keeps more
than 256 MB outside the scratch).

## Related work

[Strata](https://github.com/Niko1221/Strata) lends slots of its GPU expert
cache to the prompt path and refills them afterwards, inside its own engine.
This patch lends the dense decoder weights as well, inside SGLang, keeping the
decode graphs' addresses through torch_memory_saver.

## License

Apache-2.0, like KTransformers and SGLang.
