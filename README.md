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
valid) and streams each layer's weights back from a pinned host copy just
before that layer runs. Buffers only a prefill uses live in a second region,
the *scratch*, that is resident only during prefill. Everything is back
before the next decode step. The prefill gets several GB more, the chunk
grows, and the forward count drops.

## Who this is for

People who already run a KTransformers hybrid (kt-kernel built for their
CPU, experts on the CPU, SGLang serving attention and hot experts on the GPU)
and are comfortable patching an SGLang tree. This is not a packaged install:
each model here runs on its own SGLang tree, the expert streamer the patches
edit is not in upstream SGLang (`streamers/` has the versions they expect),
and the DeepSeek V4 zero-copy streamer also needs kt-kernel built with
shared-memory expert arenas. Once applied it needs no tuning: the chunk is
calibrated on the first launch, and a tree without the patch runs as before.

## Results

2x RTX 5070 Ti 16 GB (PCIe 4.0 x16, no P2P), dual EPYC 7452, TP=2, all CPU
experts in kt-kernel (AVX2). Prefill and decode in tokens/s, base -> lend;
base is the same server without the lend.

| Model | hot experts | chunk | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16 | 48 | 2048 -> 32768 | 1,568 -> **6,492-6,507** | 1,533 -> **6,406** | 91-97 -> 98-100 | bit-identical |
| MiMo-V2.6-Flash MXFP4 | 12 | 2048 -> 22528 | 525 -> **1,962-1,978** | 496 -> **1,550** | 30-32 -> 32 | top-20 spread unchanged ¹ |
| DeepSeek V4.1-Flash MXFP4, 1M ctx | 5 | 2048 -> 8192 | 255 -> **636-639** | 259 -> **594** | 25.4-25.6 -> 27.4-27.9 | NLL z -1.90 ¹ |
| DeepSeek V4-Flash-Vision MXFP4 | 10 -> 12 | 2048 -> 16384 | 601 -> **900-919** | 598 -> **849** | 36.5-39.9 -> 42.5-44.8 | NLL z -0.58 ¹ |
| GLM-5.3-Flash NVFP4 experts | 0 -> 10 | 2048 -> 14336 | 163 -> **326-327** | 155 -> **292** | 18.3 -> 19.9-20.1 | NLL z -0.35 / -1.04 |
| Qwen3.8-Flash-Next NVFP4 | 32 | 16384 ² | 4,123-4,145 | 4,604-4,608 (105K) | 57.8-59.4 | bit-identical |

The lend chunk is the one the calibration picked. *Bit-identical*: top-20
next-token logprobs at 312/548/1,512-token prompts match the base server bit
for bit. *NLL z*: the same token ids read teacher-forced on both servers, z
of the mean per-token log-prob difference (negative = lend side lower).
*Top-20 spread*: MiMo does not repeat bit for bit even without the lend, so
the lend side's top-20 logprobs were checked to stay within the base's
run-to-run spread, with the same top-1.

¹ Measured on an earlier lend setup (MiMo at chunk 8192, V4.1 at chunk 4096,
Vision at hot 10 and chunk 8192), not on the row's.
² Developed with the lend from the start; there is no base row.

Why the gains differ: Qwen3.5 and MiMo forwards are mostly fixed cost (4x),
V4.1 and GLM about half (2x), and Vision's forward already scales with its
rows at chunk 8192 (1.4x). Decode never got slower.

GLM-5.3 has no streamer; its full-GPU prefill writes every expert of a layer
into a 2.11 GiB slot, which used to be reserved out of the KV pool, so hot
experts and the GPU prefill excluded each other. With the slot in the scratch
both run in one server, and the KV pool still grew (198K -> 213K tokens).

## Applying it

1. Install torch_memory_saver in the server's venv (tested 0.0.9.post1; the
   lend uses its hook mode `torch`).
2. On a DeepSeek V4 tree (V4.1, V4-Flash, Vision, MiMo), first share the
   streamer's graph inputs across chunk sizes:
   `python patches/sglang-kt-stream-shared-graph-inputs.py <sglang tree>`
   (a tree that already has it is skipped).
3. Patch the tree with the script for it:

   | Tree | Patch |
   |---|---|
   | Qwen3.8-Flash-Next (SGLang v0519 fork) | `patches/sglang-kt-prefill-lend.py` |
   | DeepSeek V4 and its copies | `patches/sglang-dsv41-kt-prefill-lend.py` |
   | Qwen3.5 (upstream SGLang, bf16 streamer) | `patches/sglang-qwen35-kt-prefill-lend.py` |
   | GLM-5.3 (full-GPU prefill) | `patches/sglang-glm53-kt-prefill-lend.py` |

   `python patches/<patch> <sglang tree>` copies `kt_lend.py` into
   `layers/moe/` and edits by exact anchors. It refuses when an anchor is not
   found exactly once, and skips a file that already carries its marker.
4. Patch the installed kt-kernel, in this order:
   `python patches/kt-kernel-shared-gpu-output.py <.../kt_kernel/experts_base.py>`,
   then `python patches/kt-kernel-lend-gpu-output.py` on the same file.
   Without them kt-kernel's output buffer grows with the prefill outside the
   scratch, and the weights may not fit back.
5. Start the server with `KT_PREFILL_LEND=1` and a larger
   `--chunked-prefill-size`. To have the chunk picked for you, source
   `tools/kt-lend-auto.sh` in your launcher:

   ```bash
   source tools/kt-lend-auto.sh
   # launcher, chunk env var, port, then anything that changes the fit
   kt_lend_auto "$0" MY_CHUNK 30000 my-model ctx262144 mem0.85 hot12
   # ... --chunked-prefill-size "$KT_LEND_CHUNK"
   ```

   The first launch of a setup starts the launcher once more with
   `--background` and a 50-60K-token prompt, halving the chunk on
   out-of-memory, so the launcher must accept `--background`, print
   `Run directory: <dir>` and leave `<dir>/server.pid` and
   `<dir>/server.log`. After that the server's own measurement is reused
   from `~/.cache/kt-lend/`. Setting the chunk env var skips all of it.
   On DeepSeek V4 and MiMo the SWA pool must hold a chunk plus a page;
   `kt_lend_swa_ratio <chunk> <max_total_tokens> <ratio>` gives the ratio.

## Requirements and limits

- The expert streamer, or GLM's full-GPU prefill. Without a streamed prefill
  nothing is lent.
- Prefill must run eagerly (no prefill CUDA graphs); decode graphs are fine.
- Tested only with TP=2 on one host. Not tested: EP, DP attention, PP, a
  speculative draft model (a draft built after the target stays resident),
  SGLang's own `--enable-memory-saver`.
- A model whose layers read another layer's weights, or code outside the
  decoder that reads decoder weights during prefill, faults (illegal address)
  instead of reading stale data. `KT_PREFILL_LEND_SKIP` (default `engram`)
  keeps matching parameters resident.
- The chunk is fixed at launch; what a run measures applies to the next
  launch, which raises the chunk at most twofold. A prefill that never filled
  a chunk may only lower it.
- Where the context pool grows with the chunk (DeepSeek V4's SWA pool) a
  larger chunk costs some full-attention KV (V4.1 at chunk 8192: 1,031,680
  tokens instead of 1M).

## Knobs

- `KT_PREFILL_LEND=1` — on.
- `KT_PREFILL_LEND_SKIP` — parameter-name substrings kept resident (`engram`).
- `KT_PREFILL_LEND_MIN_BYTES` — smallest parameter lent (1 MiB).
- `KT_PREFILL_LEND_MARGIN_MB` — room left free in the measured chunk (512).
- `KT_PREFILL_LEND_IDLE` — parameter-name substrings lent but not brought
  back during a prefill, for weights the prefill never reads (GLM's hot
  experts; measured within 1%, so off by default).
- `KT_PREFILL_LEND_HINT` — where the measurement is written (set by
  `kt-lend-auto.sh`).
- `KT_PREFILL_LEND_DEBUG=1` — memory per prefill, and a torch memory snapshot
  when a prefill leaves more than 256 MB outside the scratch.

## How it is wired

For porting to another tree. Five points, none in a model file:

- `model_loader._initialize_model`: `kt_lend.bind_model(model)` right after
  the model is built. The largest `ModuleList` named `layers` is the decoder;
  per layer, every CUDA parameter of at least 1 MiB moves into one span.
- the weight-processing loop of `load_weights_and_postprocess`:
  `kt_lend.after_module(module)` puts a layer back into its span as soon as a
  repack (Marlin, swizzled scales) has moved its parameters out, so originals
  and repacked copies do not pile up.
- the end of `load_weights_and_postprocess`: `kt_lend.after_load()` lays out
  each layer again from what the load left (fused, renamed and aliased
  parameters included), mirrors the spans into pinned memory and pauses the
  scratch, before the KV pool is sized.
- `ModelRunner.forward`, before `_forward_raw`: `kt_lend.before_forward(batch)`
  switches phase. A forward of at least the streamer's threshold (GLM:
  `--kt-gpu-prefill-token-threshold`) enters prefill; the next shorter one
  returns to decode, after checking the weights fit back (it raises with the
  numbers instead of letting the remap abort).
- the prefill-only buffers: `kt_lend.scratch_empty(...)` / `scratch_region()`.

A layer is entered through its forward pre-hook, a pre-hook on each child
holding lent parameters, or any other `forward*` method of the layer (DeepSeek
V4 calls `forward_hc_pre_from_prev`); entering layer i points layer i-1 back
at its span, and layer i+1 is prefetched on a side stream.

## Related work

[Strata](https://github.com/Niko1221/Strata) lends slots of its GPU expert
cache to the prompt path and refills them afterwards, inside its own engine.
This patch lends the dense decoder weights as well, inside SGLang, keeping the
decode graphs' addresses through torch_memory_saver.

## License

Apache-2.0, like KTransformers and SGLang.
