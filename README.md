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
grows, and the forward count drops. By default the KV pool keeps every token
it would have without the lend.

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

2x RTX 5070 Ti 16 GB (PCIe 4.0 x16, no P2P), dual EPYC 7452, all CPU experts
in kt-kernel (AVX2). Prefill and decode in tokens/s, base -> lend; base is
the same server without the lend.

**TP=2, both sockets**

| Model | hot experts | chunk | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16 | 48 | 2048 -> 32768 | 1,568 -> **6,492-6,507** | 1,533 -> **6,406** | 91-97 -> 98-100 | bit-identical |
| MiMo-V2.6-Flash MXFP4 | 12 | 2048 -> 22528 | 525 -> **1,962-1,978** | 496 -> **1,550** | 30-32 -> 32 | top-20 spread unchanged ¹ |
| DeepSeek V4.1-Flash MXFP4, 1M ctx | 5 | 2048 -> 6144 | 255 -> **576** | 259 -> **596** | 25.4-25.6 -> 27.6-28.0 | NLL z -1.90 ¹ |
| DeepSeek V4-Flash-Vision MXFP4 | 10 -> 12 | 2048 -> 16384 | 601 -> **900-919** | 598 -> **849** | 36.5-39.9 -> 42.5-44.8 | NLL z -0.58 ¹ |
| GLM-5.3-Flash NVFP4 experts | 0 -> 10 | 2048 -> 14336 | 163 -> **326-327** | 155 -> **292** | 18.3 -> 19.9-20.1 | NLL z -0.35 / -1.04 |
| Qwen3.8-Flash-Next NVFP4 | 32 | 16384 ² | 4,123-4,145 | 4,604-4,608 (105K) | 57.8-59.4 | bit-identical |

V4.1 keeps its whole 1,048,576-token KV pool and prefills a 989,154-token
prompt to the end at chunk 6144 (2,433.9 s, the server's first prefill); see
[Long prompts](#long-prompts).

**One card or one socket**

| Model, setup | chunk | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16, TP=1, one card, one socket, hot 16, 131K ctx | 2048 -> 23552 | 893-894 -> **5,142-5,157** | 912 -> **4,341** | 37.4 -> 37.0-37.2 | bit-identical ³ |
| Qwen3.5-35B-A3B bf16, TP=2, one socket | 2048 -> 32768 | 821-822 -> **4,492-4,507** | 826 -> **4,967** | 82.9-84.3 -> 79.5-83.6 | bit-identical ³ |
| Qwen3.8-Flash-Next NVFP4, TP=1, one card, one socket, hot 1, 131K ctx | 2048 -> 14336 | (out of memory) -> **3,030-3,069** | **3,232** | 27.1 | bit-identical ³ |
| MiMo-V2.6-Flash MXFP4, TP=1, one card, one socket, hot 0, 131K ctx | 2048 -> 16384 ⁴ | 151 -> **1,098-1,104** | (over the KV pool) | 12.5-12.8 -> 12.4-12.6 | bit-identical where the chunking matches ³ |

On one 16 GB card the lend is what makes room for the prefill at all:
without it Flash-Next ran out of memory on its first 8K prefill at the same
memory fraction, and MiMo needed a fraction of 0.80, a 59,523-token KV pool
and 1,188-row batches (its SWA pool held 1,190 tokens) where the lend kept
94,533 tokens and 12.4K-row batches.

The lend chunk is the one the calibration picked. *Bit-identical*: top-20
next-token logprobs at 312/548/1,512-token prompts match the base server bit
for bit. *NLL z*: the same token ids read teacher-forced on both servers, z
of the mean per-token log-prob difference (negative = lend side lower).
*Top-20 spread*: MiMo at TP=2 does not repeat bit for bit even without the
lend, so the lend side's top-20 logprobs were checked to stay within the
base's run-to-run spread, with the same top-1.

¹ Measured on an earlier lend setup (MiMo at chunk 8192, V4.1 at chunk 4096,
Vision at hot 10 and chunk 8192), not on the row's.
² Developed with the lend from the start; there is no base row.
³ Also bit-identical to itself after twelve prompts of different lengths
with `KT_STREAM_GRAPH_KEEP=2`, which drops and re-captures the streamer's
CUDA graphs. MiMo at 1,725 tokens: the base server split that prompt
1,188 + 537 (its small SWA pool) and differs; at 358 and 634 tokens, where
both run one chunk, they match.
⁴ Batches held at 12.4K rows by the SWA pool; see `KT_PREFILL_LEND_SWA_FIT`.

Why the gains differ: Qwen3.5 and MiMo forwards are mostly fixed cost (4x),
V4.1 and GLM about half (2x), and Vision's forward already scales with its
rows at chunk 8192 (1.4x). Decode never got slower.

GLM-5.3 has no streamer; its full-GPU prefill writes every expert of a layer
into a 2.11 GiB slot, which used to be reserved out of the KV pool, so hot
experts and the GPU prefill excluded each other. With the slot in the scratch
both run in one server, and the KV pool still grew (198K -> 213K tokens).

## Long prompts

A chunk's memory grows with the prefix on some models (DeepSeek V4's indexer
and page tables are sized by it), so the chunk must hold at the longest
prompt the server takes, not at the calibration prompt. On a DeepSeek V4 tree
the calibration measures it there with the *probe*
(`patches/sglang-kt-lend-probe.py`): before the calibration server starts
serving, it runs a few chunks over prefixes at 0, half the context and the
context's end whose KV is allocated but never computed. A real 1M-token
prefill takes 40 minutes; the probe takes two. V4.1 at 6144 rows, peak above
the prefill baseline:

| prefix | probe, reserved | probe, allocated | real prefill, reserved |
|---:|---:|---:|---:|
| 131K | 3,168 MiB | 2,098 MiB | 3,252-3,292 (38K) |
| 524K | 3,812 | 2,317 | 3,146-3,270 (510K) |
| 991K | 5,286 | 2,921 | 6,272 (989K) |

The calibration hinted 9216, 8192 and 6144 at the three prefixes and kept
6144. Reserved runs well above allocated at long prefixes: each chunk's
prefix-sized buffers are a little larger than the last's and do not fit the
freed ones. The lend hands torch's unused segments back before every chunk
past 64K tokens, which removes most of it between forwards (2048 rows at
993K: 3,028 -> 994 MiB of the gap) but not inside one; torch's
`expandable_segments` would, and torch_memory_saver does not support it yet.

## Applying it

1. Install torch_memory_saver in the server's venv (tested 0.0.9.post1; the
   lend uses its hook mode `torch`).
2. On a tree with the expert streamer, keep its CUDA graphs bounded:
   `python patches/sglang-kt-stream-graph-keep.py <sglang tree>`. The
   streamer captured one graph per row count and kept them all; a prompt's
   last chunk brings a new row count, so the device memory outside torch rose
   12-14 MiB per prompt of a new length until the weights could not come back.
3. On a DeepSeek V4 tree (V4.1, V4-Flash, Vision, MiMo), share the streamer's
   graph inputs across chunk sizes:
   `python patches/sglang-kt-stream-shared-graph-inputs.py <sglang tree>`.
4. Patch the tree with the script for it:

   | Tree | Patch |
   |---|---|
   | Qwen3.8-Flash-Next (SGLang v0519 fork) | `patches/sglang-kt-prefill-lend.py` |
   | DeepSeek V4 and its copies | `patches/sglang-dsv41-kt-prefill-lend.py` |
   | Qwen3.5 (upstream SGLang, bf16 streamer) | `patches/sglang-qwen35-kt-prefill-lend.py` |
   | GLM-5.3 (full-GPU prefill) | `patches/sglang-glm53-kt-prefill-lend.py` |

   `python patches/<patch> <sglang tree>` copies `kt_lend.py` into
   `layers/moe/` and edits by exact anchors. It refuses when an anchor is not
   found exactly once, and skips a file that already carries its marker.
5. On a DeepSeek V4 tree, the probe:
   `python patches/sglang-kt-lend-probe.py <sglang tree>`.
6. Patch the installed kt-kernel, in this order:
   `python patches/kt-kernel-shared-gpu-output.py <.../kt_kernel/experts_base.py>`,
   then `python patches/kt-kernel-lend-gpu-output.py` on the same file.
   Without them kt-kernel's output buffer grows with the prefill outside the
   scratch, and the weights may not fit back.
7. Start the server with `KT_PREFILL_LEND=1` and a larger
   `--chunked-prefill-size`. To have the chunk picked for you, source
   `tools/kt-lend-auto.sh` in your launcher:

   ```bash
   source tools/kt-lend-auto.sh
   export KT_PREFILL_LEND_KV_CAP="$MAX_TOTAL_TOKENS"       # the KV guard
   export KT_PREFILL_LEND_PROBE_CONTEXT="$MAX_TOTAL_TOKENS" # V4 tree: calibrate with the probe
   # launcher, chunk env var, port, then anything that changes the fit
   kt_lend_auto "$0" MY_CHUNK 30000 my-model ctx262144 mem0.85 hot12
   # ... --chunked-prefill-size "$KT_LEND_CHUNK"
   ```

   The first launch of a setup starts the launcher once more with
   `--background` to calibrate, so the launcher must accept `--background`,
   print `Run directory: <dir>` and leave `<dir>/server.pid` and
   `<dir>/server.log`. With `KT_PREFILL_LEND_PROBE_CONTEXT` that server runs
   the probe; without it, it gets a 50-60K-token prompt. Out of memory halves
   the chunk. After that the server's own measurement is reused from
   `~/.cache/kt-lend/`. Setting the chunk env var skips all of it.

## The KV pool

A wider chunk needs no KV on most models, but where the SWA pool must hold
the chunk (DeepSeek V4, MiMo) it widens with it, and that comes out of the
full pool. The lend takes nothing from the KV pool by default:

- The KV guard (`KT_PREFILL_LEND_KV_CAP`, the `max_total_tokens` a launcher
  asks for): when the last server of a setup got a smaller full pool than
  that, the chunk it ran is remembered as one that cuts the KV pool and the
  next launch goes 1024 below it. Where even chunk 2048 does not reach the
  cap (memory, not the lend), the full pool at 2048 becomes the reference.
- On V4.1 a chunk of 10240 cut the full pool from 1,048,576 to 896,000
  (114K prompts 790 tok/s instead of about 600), so its launcher stays at
  6144. The SWA cap counts two chunks in flight, while the overlap scheduler
  still holds the previous chunk's slots, so every other batch came in at
  3072 rows; the launcher adds prefix tails to fill them only as far as the
  pool's slack pays (13 tails at 6144, budget 1,049,856 for a cap of
  1,048,576).
- `kt_lend_swa_ratio` (ratio-sized SWA pools: V4-Flash, Vision, MiMo)
  computes the ratio from the cap. `KT_PREFILL_LEND_SWA_FIT=1` computes it
  from the full pool an earlier launch got instead, which trades KV for
  chunk: MiMo on one card went from 12.1K- to 13.9K-row batches (40K prompts
  1,107 -> 1,331 tok/s) and from a 97,699- to an 81,552-token full pool.

## Requirements and limits

- The expert streamer, or GLM's full-GPU prefill. Without a streamed prefill
  nothing is lent.
- Prefill must run eagerly (no prefill CUDA graphs); decode graphs are fine.
- Tested on one host: TP=2 over two sockets, TP=2 on one socket, TP=1 on one
  card and one socket. Not tested: EP, DP attention, PP, a speculative draft
  model (a draft built after the target stays resident), SGLang's own
  `--enable-memory-saver`.
- A model whose layers read another layer's weights, or code outside the
  decoder that reads decoder weights during prefill, faults (illegal address)
  instead of reading stale data. `KT_PREFILL_LEND_SKIP` (default `engram`)
  keeps matching parameters resident.
- The chunk is fixed at launch; what a run measures applies to the next
  launch, which raises the chunk at most twofold. A window of fewer rows than
  the chunk only lowers it.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is refused by
  torch_memory_saver.
- The Flash-Next tree's GPU expert path does not run with 0 hot experts (with
  or without the lend); use at least 1.

## Knobs

- `KT_PREFILL_LEND=1` — on.
- `KT_PREFILL_LEND_SKIP` — parameter-name substrings kept resident (`engram`).
- `KT_PREFILL_LEND_MIN_BYTES` — smallest parameter lent (1 MiB).
- `KT_PREFILL_LEND_MARGIN_MB` — room left free in the measured chunk (256;
  what a window adds outside torch's allocator measured 0-62 MiB).
- `KT_PREFILL_LEND_LONG_SEQ` — from this sequence length (65536) an extend
  below the streaming threshold still runs lent (a long prompt's last chunk
  makes prefix-sized buffers a card with its weights back has no room for),
  and torch's cache is emptied before each chunk.
- `KT_PREFILL_LEND_IDLE` — parameter-name substrings lent but not brought
  back during a prefill, for weights the prefill never reads (GLM's hot
  experts; measured within 1%, so off by default). Turns `LONG_SEQ`'s first
  part off.
- `KT_PREFILL_LEND_HINT` — where the measurement is written (set by
  `kt-lend-auto.sh`).
- `KT_PREFILL_LEND_KV_CAP`, `KT_PREFILL_LEND_PROBE_CONTEXT`,
  `KT_PREFILL_LEND_MAX_CHUNK` (widest chunk the calibration starts at),
  `KT_PREFILL_LEND_SWA_FIT` — `kt-lend-auto.sh`, above.
- `KT_PREFILL_LEND_PROBE="<chunks>@<prefixes>"`, `KT_PREFILL_LEND_PROBE_STEPS`
  (4 chunks a point) — run the probe directly.
- `KT_STREAM_GRAPH_KEEP` — streamer CUDA graphs kept for row counts other
  than the largest (8).
- `KT_PREFILL_LEND_DEBUG=1` — memory per prefill (reserved and allocated, and
  how much more the device held outside torch at the end), and a torch memory
  snapshot when a prefill leaves more than 256 MB outside the scratch.

## How it is wired

For porting to another tree. Five points, none in a model file:

- `model_loader._initialize_model`: `kt_lend.bind_model(model)` right after
  the model is built. The largest `ModuleList` named `layers` is the decoder;
  per layer, every CUDA parameter of at least 1 MiB moves into one span.
- the weight-processing loop of `load_weights_and_postprocess`:
  `kt_lend.after_module(module)` puts a layer back into its span as soon as a
  repack (Marlin, swizzled scales, Flash-Next's fused in_proj) has moved its
  parameters out, so originals and repacked copies do not pile up.
- the end of `load_weights_and_postprocess`: `kt_lend.after_load()` lays out
  each layer again from what the load left (fused, renamed and aliased
  parameters included), mirrors the spans into pinned memory and pauses the
  scratch, before the KV pool is sized.
- `ModelRunner.forward`, before `_forward_raw`: `kt_lend.before_forward(batch)`
  switches phase. A forward of at least the streamer's threshold (GLM:
  `--kt-gpu-prefill-token-threshold`), or an extend over a long sequence,
  enters prefill; the next other one returns to decode (`end_window()`),
  after checking the weights fit back (it raises with the numbers instead of
  letting the remap abort).
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
