# Borrowed Prefill

**During a prefill the GPU's dense weights sit idle. Borrow their VRAM for the
prefill chunk, give it back before decode: a KTransformers + SGLang hybrid
prefills 3-4x faster on the same card, with the same KV pool.**

Qwen3.8-Flash-Next (NVFP4 MoE, 2x RTX 5070 Ti 16 GB, dual EPYC 7452):

| | without | with Borrowed Prefill |
|---|---|---|
| 38K-token prompt | 1,337 tok/s | **4,154 tok/s** |
| 105K-token prompt | 1,367 tok/s | **4,596 tok/s** |
| decode | 58.5 tok/s | 59.3 tok/s |
| KV pool | 231,616 tokens | 262,144 tokens |
| output | | bit-identical |

Same prompts, same server; the only change is the lend (the project's name
for the mechanism; `kt-prefill-lend` is the repo). Five more models, including
DeepSeek V4.1 at a 1M-token context, are in [Results](#results).

## How it works

In a KTransformers hybrid (CPU experts, GPU attention and hot experts), a long
prompt is prefilled by streaming the CPU-resident experts over PCIe into the
GPU, chunk by chunk. Every forward pays a fixed transfer cost, so a larger
chunk is faster, but on a 16 GB card the chunk is capped by what the decode
footprint (attention weights, hot experts, KV pool, CUDA graphs) leaves free.

During such a prefill the lend pauses the decoder-layer weights with
[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) (physical
memory released, virtual addresses kept, so the decode CUDA graphs stay
valid) and streams each layer's weights back from a pinned host copy just
before that layer runs. Buffers only a prefill uses live in a second region,
the *scratch*, that is resident only during prefill. Everything is back
before the next decode step. The prefill gets several GB more, the chunk
grows, and the forward count drops. The KV pool keeps every token it would
have without the lend.

## Serving a model

The repo serves from a *profile*: `profiles/<name>.json` holds the SGLang
command and environment of a measured launch with every host- and size-
dependent value as a placeholder (`{config_dir}`, `{hot}`, `{context}`,
`{placement}`, ...), the defaults it ran with per TP, and the VRAM constants
calibrated from its log. `tools/serve.py` fills a profile in from the host
and the model and starts the server:

```sh
tools/serve.py --profile <name> --venv <venv> --tree <patched sglang tree> --config-dir <model dir> \
               [--gguf DIR] [--draft-dir DIR] [--placement FILE] [--tp 1|2] [--context T] [--hot N] [--dry-run]
```

It probes the host (`hostprobe.py`: GPUs by NUMA node, cores and RAM per
node, THP / swap / `min_free_kbytes`), reads the model (`modelprobe.py`:
experts and their slot size, dense bytes, the KV family and its bytes per
token, all from the weight headers), sizes the launch (`plan-launch.py`: the
per-rank VRAM budget spent in order — dense, lm_head, draft, the KV pool for
the context, linear-attention states, graphs, workspace — and the hot experts
per layer that remain, minus what the empty pool lends), binds each rank to
its socket, and prints the command with the arithmetic. `--dry-run` prints
without starting; `--hot` / `--chunk` / `--mem-fraction` override the plan.

Profiles in the repo, each verified by the same prompt-NLL check (three fixed
texts scored teacher-forced on the in-house launcher and on `serve.py`,
identical to four decimals):

| profile | model | TP | KV | pool |
|---|---|---|---|---|
| `flash-next-gguf` | Qwen3.8-Flash-Next from Strata's IQ3_S GGUF, MTP draft | 1 / 2 | bf16 | 65,536 / 121,152 |
| `flash-next-nvfp4` | Qwen3.8-Flash-Next NVFP4 (ModelOpt) | 2 | bf16 | 262,144 |
| `glm53-flash` | GLM-5.3-Flash, NVFP4 experts, full-GPU prefill | 2 | bf16 | 131,072 |
| `dsv4-flash-vision` | DeepSeek V4-Flash-Vision MXFP4, 1M context | 2 | fp8 | 262,144 |
| `mimo-v26-flash` | MiMo-V2.6-Flash MXFP4 | 2 | fp8 | 262,144 |

**A new model** takes one launch of whatever launcher already runs it:

```sh
tools/profile-from-launch.py --pid <launch_server pid> --name <name> --out profiles/<name>.json \
                             --config-dir <model dir> [--experts DIR] [--draft-dir DIR] --cache-dir <dir>
tools/plan-calibrate.py <that launch's server.log> --hot N --slot-bytes B --moe-layers L \
                        --vram-total BYTES --mem-fraction F [--kv-dtype fp8] --write profiles/<name>.json
```

The first reads the running process's environment and command line and turns
the paths and sized values into placeholders (nothing is transcribed from a
script). The second reads the log's accounting lines (weights, KV pool,
states, graphs; DeepSeek V4's hybrid pool lines included) and writes the
constants the plan then uses: with them, the plan predicts the pool and the
hot count of a launch at another memory fraction or context without a second
launch. Without a profile, `plan-launch.py` still sizes a launch from the
headers alone; its constants are then the ones measured on our cards and are
printed with every plan.

What stays model-specific: a hot-expert placement file (`--placement`, a JSON
of the hottest experts per layer from an earlier run). Without one the serve
runs the loader's trivial placement, which is valid but a different set of
GPU experts, so its NLL differs from a placed launch's.

### From nothing

Flash-Next from a GGUF, the whole serve (both patched trees, a venv with the
measured versions, kt-kernel built for your CPU and GPU, the model
directories, a sized launch):

```sh
tools/install.sh ~/kt-serve                                                 # clone, patch, venv, build (10-20 min)
tools/make-config-dir.py <HF dir> <gguf dir> ~/kt-serve/config               # config + tokenizer, quant_method gguf
~/kt-serve/venv/bin/python tools/make-draft-dir.py <mtp gguf> <HF dir> ~/kt-serve/draft   # the MTP draft (optional)
tools/serve.py --profile flash-next-gguf --venv ~/kt-serve/venv --tree ~/kt-serve/sglang \
               --config-dir ~/kt-serve/config --gguf <gguf dir> --draft-dir ~/kt-serve/draft --tp 1 --context 65536
```

The HF dir is the checkpoint's config and tokenizer; the GGUF dir is Strata's
IQ3_S set or any Flash-Next GGUF with expert types the CPU kernels have
(`trees/kt-kernel/`). `trees/flash-next-gguf/` is the serving tree as one
patch on upstream `release/v0.5.19` (the streamer, the lend, the hot cache,
the KV lend, the GGUF loader); `trees/glm53/` the same for GLM-5.3.

### On a server you already run

```sh
pip install torch_memory_saver==0.0.9.post1          # in the server's venv
python patches/sglang-kt-stream-graph-keep.py <sglang tree>
python patches/sglang-kt-stream-shared-graph-inputs.py <sglang tree>   # DeepSeek V4 trees only
python patches/sglang-kt-prefill-lend.py <sglang tree>                 # or the patch for your tree, see Applying it
KT_PREFILL_LEND=1 <your launcher>                                      # first launch calibrates the chunk, ~2 min
```

Then send a long prompt and watch `nvidia-smi`: the usage climbs for the
prefill and drops back for decode. `tools/kt-lend-auto.sh` has the launcher
side (calibration, the KV guard); `KT_PREFILL_LEND=0` turns it off.

## Who this is for

People who run a KTransformers hybrid (kt-kernel built for their CPU, experts
on the CPU, SGLang serving attention and hot experts on the GPU) and are
comfortable patching an SGLang tree. Each model here runs on its own SGLang
tree, the expert streamer the patches edit is not in upstream SGLang
(`streamers/` has the versions they expect), and the DeepSeek V4 zero-copy
streamer also needs kt-kernel built with shared-memory expert arenas. Once
applied it needs no tuning: the chunk is calibrated on the first launch, and
a tree without the patch runs as before.

## Results

2x RTX 5070 Ti 16 GB (PCIe 4.0 x16, no P2P), dual EPYC 7452, all CPU experts
in kt-kernel (AVX2). Prefill and decode in tokens/s, base -> lend; base is
the same server without the lend.

**TP=2, both sockets**

| Model | hot experts | chunk | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | 32 | 2048 -> 16384 | 1,336-1,337 -> **4,153-4,154** | 1,367 -> **4,596** (105K) | 58.5-58.7 -> 54.8-59.3 | bit-identical |
| Qwen3.5-35B-A3B bf16 | 48 | 2048 -> 32768 | 1,568 -> **6,492-6,507** | 1,533 -> **6,406** | 91-97 -> 98-100 | bit-identical |
| MiMo-V2.6-Flash MXFP4 | 12 | 2048 -> 22528 | 525 -> **1,962-1,978** | 496 -> **1,550** | 30-32 -> 32 | top-20 spread unchanged |
| DeepSeek V4.1-Flash MXFP4, 1M ctx | 5 | 2048 -> 6144 | 255 -> **576** | 259 -> **596** | 25.4-25.6 -> 27.6-28.0 | NLL z -1.90 |
| DeepSeek V4-Flash-Vision MXFP4 | 10 -> 12 | 2048 -> 16384 | 601 -> **900-919** | 598 -> **849** | 36.5-39.9 -> 42.5-44.8 | NLL z -0.58 |
| GLM-5.3-Flash NVFP4 experts | 0 -> 10 | 2048 -> 14336 | 163 -> **326-327** | 155 -> **292** | 18.3 -> 19.9-20.1 | NLL z -0.35 / -1.04 |

V4.1 keeps its whole 1,048,576-token KV pool and prefills a 989,154-token
prompt to the end at chunk 6144 (406 tokens/s, the server's first prefill); see
[Long prompts](#long-prompts).

**One card or one socket**

| Model, setup | chunk | 38K prompt | 114K prompt | decode | quality vs base |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16, TP=1, one card, one socket, hot 16, 131K ctx | 2048 -> 23552 | 893-894 -> **5,142-5,157** | 912 -> **4,341** | 37.4 -> 37.0-37.2 | bit-identical |
| Qwen3.5-35B-A3B bf16, TP=2, one socket | 2048 -> 32768 | 821-822 -> **4,492-4,507** | 826 -> **4,967** | 82.9-84.3 -> 79.5-83.6 | bit-identical |
| Qwen3.8-Flash-Next NVFP4, TP=1, one card, one socket, hot 1, 131K ctx | 2048 -> 14336 | (out of memory) -> **3,030-3,069** | **3,232** | 27.1 | bit-identical |
| MiMo-V2.6-Flash MXFP4, TP=1, one card, one socket, hot 0, 131K ctx | 2048 -> 16384 | 151 -> **1,098-1,104** | (over the KV pool) | 12.5-12.8 -> 12.4-12.6 | bit-identical where the chunking matches |

On one 16 GB card the lend is what makes room for the prefill at all:
without it Flash-Next ran out of memory on its first 8K prefill at the same
memory fraction, and MiMo needed a fraction of 0.80, a 59,523-token KV pool
and 1,188-row batches (its SWA pool held 1,190 tokens) where the lend kept
94,533 tokens and 12.4K-row batches.

The lend chunk is the one the calibration picked. *Bit-identical*: top-20
next-token logprobs at 312/548/1,512-token prompts match the base server bit
for bit, also after twelve prompts of different lengths with
`KT_STREAM_GRAPH_KEEP=2`, which drops and re-captures the streamer's CUDA
graphs. *NLL z*: the same token ids read teacher-forced on both servers, z of
the mean per-token log-prob difference (negative = lend side lower). *Top-20
spread*: MiMo at TP=2 does not repeat bit for bit even without the lend, so
the lend side's top-20 logprobs were checked to stay within the base's
run-to-run spread, with the same top-1.

Why the gains differ: Qwen3.5 and MiMo forwards are mostly fixed cost (4x),
V4.1 and GLM about half (2x), and Vision's forward already scales with its
rows (1.4x). Decode never got slower.

GLM-5.3 has no streamer; its full-GPU prefill writes every expert of a layer
into a 2.11 GiB slot that lives in the scratch, so hot experts and the GPU
prefill run in one server and the KV pool still grows (198K -> 213K tokens).

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
6144. Reserved runs above allocated at long prefixes: each chunk's
prefix-sized buffers are a little larger than the last's and do not fit the
freed ones. The lend hands torch's unused segments back before every chunk
past 64K tokens, which removes most of it between forwards but not inside
one; torch's `expandable_segments` would, and torch_memory_saver does not
support it.

## Applying it

1. Install torch_memory_saver in the server's venv (tested 0.0.9.post1; the
   lend uses its hook mode `torch`).
2. On a tree with the expert streamer, keep its CUDA graphs bounded:
   `python patches/sglang-kt-stream-graph-keep.py <sglang tree>`. The
   streamer captures one graph per row count; a prompt's last chunk brings a
   new row count, and unbounded graphs cost 12-14 MiB of device memory per
   prompt of a new length until the weights cannot come back.
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
- On V4.1 a chunk of 10240 cuts the full pool from 1,048,576 to 896,000, so
  its launcher stays at 6144. The SWA cap counts two chunks in flight, while
  the overlap scheduler still holds the previous chunk's slots, so every other
  batch comes in at 3072 rows; the launcher adds prefix tails to fill them
  only as far as the pool's slack pays (13 tails at 6144, budget 1,049,856
  for a cap of 1,048,576).
- With the plan (next section) the SWA pool no longer sets the chunk: it is
  budgeted at a small chunk and widened outside the KV budget, lent outside
  prefill.
- `kt_lend_swa_ratio` (ratio-sized SWA pools: V4-Flash, Vision, MiMo)
  computes the ratio from the cap. `KT_PREFILL_LEND_SWA_FIT=1` computes it
  from the full pool an earlier launch got instead, which trades KV for
  chunk: MiMo on one card goes from 12.1K- to 13.9K-row batches (40K prompts
  1,107 -> 1,331 tok/s) and from a 97,699- to an 81,552-token full pool.

## The chunk per request: the KV pool's free pages and the plan

The lend sizes one chunk at launch. With `--plan` (serve.py) or
`kt_prefill_plan_setup` (kt-lend-auto.sh) the chunk is planned per request
instead, and a prefill also borrows the KV pool's *free* pages. Four pieces,
in `kv-vmm/` and `patches/`, none model-specific:

- **The KV pool on remappable memory** (`KT_KV_VMM=1`, `kv_vmm.c`,
  `kt_kv_vmm.py`). Every buffer SGLang makes in its `kv_cache` region is a
  CUDA virtual-memory range of 8 MiB chunks (the device granularity for
  buffers under 64 MiB) that can be unmapped and mapped again, zeroed, at the
  same addresses, so the decode CUDA graphs that read it stay valid. On torch
  2.13 the region is a `CUDAPluggableAllocator` MemPool; on older torch, whose
  MemPool fails there, the region makes the tensors itself.
- **The loan** (`KT_PREFILL_KV_LEND=1`, `kt_kv_prefill_lend.py`). When a
  prefill window opens, the free pages beyond the request's remaining prompt
  and 4,096 tokens leave the allocator's free list and every chunk holding only
  such pages is unmapped; at the window's end they are mapped back, zeroed,
  before the weights. The KV always wins: the pool keeps its size,
  `available_size` counts the lent pages, and an allocation that needs them
  maps them back first. A pool split into sub-pools by page (DeepSeek V4's
  per-compress-ratio and indexer pools, whose pages line up with the full
  pool's) lends through all of them.
- **The plan** (`KT_PREFILL_PLAN=1`, `kt_prefill_plan.py`). The launch chunk
  becomes a cap (`--plan-cap`, 36,864) and each request gets the fewest chunks
  its uncached prompt needs, at the smallest equal size, from the free memory
  every TP rank has now (a MIN all-reduce, so every rank plans the same) and a
  cost per row fitted from the lend's per-forward records
  (`tools/kt-prefill-plan-constants.py`; until enough are recorded, the lend's
  own reserved bytes a row). Until a launch's first window has measured the
  step from decode to a window, a request gets the lend's own chunk.
- **Row blocks** (`KT_GPU_STREAM_ROW_BLOCK=4096`,
  `patches/sglang-kt-stream-row-block.py` for the NVFP4 / GGUF streamers,
  `patches/sglang-kt-stream-row-block-dsv41.py` for the DeepSeek V4 one). The
  streamer's and the resident experts' CUTLASS MoE run over blocks of rows,
  so its workspace is a block's, not the chunk's (it was 104-130 KB a row).

A hybrid sliding-window model (MiMo, DeepSeek V4) must hold a whole chunk in
its SWA pool, which on those models costs several full-pool tokens a slot.
With the plan (`KT_PREFILL_SWA_LEND=1`, set for such profiles):

- the pools are budgeted at a 2,048-token chunk; the SWA pool alone is widened
  for the cap (and the DeepSeek V4 compress state that follows its pages), its
  widened part made with nothing mapped;
- outside prefill its free pages beyond 4,096 tokens are unmapped, and what a
  prefix cache keeps beyond the pool the lend's own chunk would have had is
  handed back through the cache's own eviction (`tree_cache.evict`);
- SGLang's chunk-cache SWA eviction counts the chunk in flight as the one
  planned for the request (`managers/schedule_batch.py`), so a request does
  not hold every chunk's SWA until it ends.

What SGLang itself sizes from the launch chunk is kept at the lend's chunk: the
pools' budget, and KT's staging buffer for CPU experts
(`patches/sglang-kt-staging-plan.py`, GLM-5.3's tree), which under the plan
holds the batches under the GPU prefill threshold only and gives a larger one
its own buffer.

**Results.** The same host, one launch each with and without the plan (`tools/serve.py
--plan` is the same switch), a 35K-token prompt prefilled into an empty pool,
a short turn on it, then a 35K prompt of other tokens with the first still
cached. Prefill and decode in tokens/s (the 35K prompts are 35,000 tokens):

| Model | KV pool | 35K, empty pool | 35K again, the first cached | decode | quality vs without |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | 262,144 = | 4,736 -> **5,452** | 4,834 -> 4,808 | 60.8 -> 66.1 | bit-identical |
| Qwen3.5-35B-A3B bf16 | 262,144 = | 6,446 -> **8,178** | 6,809 -> **8,883** | 99.7 -> 102.3 | bit-identical |
| DeepSeek V4.1-Flash, 1M ctx | 1,027,072 -> 1,048,576 | 452 -> **603** | 609 -> **787** | 29.5 -> 28.4 | NLL z +0.49 |
| DeepSeek V4-Flash-Vision | 262,144 = | (first use) | 839 -> 857 | 43.6 -> 42.6 | NLL z -0.07 |
| MiMo-V2.6-Flash | 262,144 = | (first use) | 1,915 -> 1,834 | 31.7 -> 32.1 | NLL z +0.90 |
| GLM-5.3-Flash NVFP4 experts | 201,280 -> **217,408** | 320 -> 321 | 322 -> 323 | 20.3 -> 20.4 | bit-identical |
| Qwen3.8-Flash-Next GGUF (IQ3_S) | 262,144 = | (first use) | 4,636 -> 4,391 | 89.6 -> 88.5 | bit-identical |

The short turn (200 tokens on the cached 35K, 64 out) is a whole-turn time, not
a prefill rate: 2.26 -> 2.08 s, 6.72 -> 5.52, 60.08 -> 47.19, 4.76 -> 4.77,
4.39 -> 4.52, 112.45 -> 111.89 and 1.44 -> 1.65 in the table's order.

No run ran out of memory. The plan does not touch the decode path; decode moves from launch to launch on this host (its inlet temperature), Flash-Next's 60.8 included. *(first use)*: the streamer's one-time setup of the
expert arenas (5-8 minutes on this host) fell into one side's first long
prompt. Where the chunk count drops the prefill gets faster (Flash-Next 2 -> 1
chunks on an empty pool, Qwen3.5 2 -> 1, V4.1 6 -> 3 and 6 -> 4 again); with a prompt already
cached there are fewer free pages to borrow and the plan takes more chunks
(Flash-Next's second 35K: 2 chunks again). Vision, MiMo and GLM keep their
chunk count at 35K (their prefill costs 290-520 KB a row reserved, and Vision
and MiMo also need the prompt's SWA mapped). GLM's pool grows because its
staging buffer no longer follows the launch chunk. V4.1's pool without the
plan is 1,027,072 here because its launcher's prefix-tail count read the
previous launch's log.

**The pages in use stay as they were.** `KT_PREFILL_KV_LEND_VERIFY=1` hashes
every page in use when a window lends (not the window's own batch's) and again
after it is mapped back; `KT_PREFILL_KV_LEND_FORCE=1` makes every window lend.
`tools/kv-lend-cache-check.py` fills the pool with cached prompts around
lending windows (seven 24K-token prompts in a 262K pool):

| model | pages in use when a window lent (most) | changed |
|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | 3,747 of 4,096 (64 tokens each) | 0 in 41 windows |
| MiMo-V2.6-Flash | 204,884 of 262,144 (1 token) | 0 in 9 windows |
| DeepSeek V4-Flash-Vision | 794 of 1,024 (256 tokens) | 0 in 9 windows |

V4.1, Qwen3.5 and GLM-5.3 have no prefix cache in these setups, so nothing else
is in their pool when a window lends; their NLL is the check. Comparing cached
continuations does not judge this on every model: Qwen3.8-Flash-Next's cached
path gives a different continuation each time the same prompt is read, with
the plan off as well (a known issue of its prefix cache, not of the lend).

**Applying it.** Build the library (`kv-vmm/build.sh`), then on the patched
tree `python kv-vmm/sglang-kt-kv-vmm.py <sglang tree>` (the modules and the
library into `layers/moe/`, the `kv_cache` region, the scheduler's chunk and
prompt-length hooks, and where the tree has them the KV cache configurator's
SWA hunks and the chunk-cache SWA eviction), the row-block patch for its
streamer, and on GLM-5.3's tree `patches/sglang-kt-staging-plan.py`. Then
`tools/serve.py ... --plan`, or in a launcher, after `kt_lend_auto`:

```bash
kt_prefill_plan_setup        # "swa" as its argument for a hybrid SWA model
# ... --chunked-prefill-size "$KT_LEND_CHUNK"
```

A setup's first launch has no measurement yet: it runs the lend at its chunk
with the loan, and records; the next one plans.

**Knobs.** `KT_KV_VMM`, `KT_PREFILL_KV_LEND`, `KT_PREFILL_PLAN`,
`KT_PREFILL_SWA_LEND` (on); `KT_PREFILL_PLAN_CAP` (36,864),
`KT_PREFILL_PLAN_SAFE` (the lend's chunk), `KT_PREFILL_PLAN_MARGIN_MB` (128),
`KT_PREFILL_PLAN_COST` / `_ROW_KB` / `_FREE_MB` / `_DELTA_MB` (from the
records); `KT_PREFILL_KV_LEND_RESERVE` (4,096 tokens kept free beyond the
prompt); `KT_PREFILL_SWA_TOKENS`, `KT_PREFILL_SWA_SIZING_CHUNK` (2,048),
`KT_PREFILL_SWA_LEND_KEEP` (4,096), `KT_PREFILL_SWA_CHUNKS_IN_FLIGHT` (from
SGLang's overlap scheduler: 2, else 1); `KT_GPU_STREAM_ROW_BLOCK` (4,096);
`KT_KV_VMM_LIB`, `KV_VMM_CHUNK_MB` (8); `KT_PREFILL_KV_LEND_VERIFY`,
`KT_PREFILL_KV_LEND_FORCE` (checks).

## Requirements and limits

- The expert streamer, or GLM's full-GPU prefill. Without a streamed prefill
  nothing is lent.
- Prefill must run eagerly (no prefill CUDA graphs); decode graphs are fine.
- Tested on one host: TP=2 over two sockets, TP=2 on one socket, TP=1 on one
  card and one socket. Not tested: EP, DP attention, PP, SGLang's own
  `--enable-memory-saver`. A speculative draft built after the target stays
  resident.
- A model whose layers read another layer's weights, or code outside the
  decoder that reads decoder weights during prefill, faults (illegal address)
  instead of reading stale data. `KT_PREFILL_LEND_SKIP` (default `engram`)
  keeps matching parameters resident.
- Without the plan the chunk is fixed at launch; what a run measures applies
  to the next launch, which raises the chunk at most twofold. A window of fewer
  rows than the chunk only lowers it.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is refused by
  torch_memory_saver.
- An extend below the streaming threshold runs with the weights back unless
  its sequence reaches `KT_PREFILL_LEND_LONG_SEQ`: a prompt's last chunk, or a
  short turn on a long conversation. The pool and the free memory are sized
  for decode, so check such an extend on a new model: a prompt of two chunks
  plus a ~160-token tail, and a few-token turn on a long conversation.
- The Flash-Next tree's GPU expert path does not run with 0 hot experts (with
  or without the lend); use at least 1.
- The plan's KV loan needs a pool allocator with one free list (SGLang's paged
  and token allocators, and an SWA allocator's two sides); others run without
  the loan. The SWA lend needs the KV cache configurator of SGLang's newer
  trees.

## Knobs

- `KT_PREFILL_LEND=1` — on.
- `KT_PREFILL_LEND_SKIP` — parameter-name substrings kept resident (`engram`).
- `KT_PREFILL_LEND_MIN_BYTES` — smallest parameter lent (1 MiB).
- `KT_PREFILL_LEND_MARGIN_MB` — room left free in the measured chunk (256;
  what a window adds outside torch's allocator measured 0-62 MiB).
- `KT_PREFILL_LEND_LONG_SEQ` — from this sequence length (65536) an extend
  below the streaming threshold still runs lent (a long prompt's last chunk
  makes prefix-sized buffers a card with its weights back has no room for),
  and torch's cache is emptied before each chunk. 65536 is sized for
  DeepSeek V4.1, whose short extends only outgrow the card at prefixes near
  1M. A model whose short extend is expensive needs it at the chunk: GLM-5.3
  EXL3 on 16 GB cards runs 8192, because its EXL3 linears rebuild fp16
  weights above 144 rows and the ~160-row tail of a 16.5K prompt ran out of
  memory with the weights back.
- `KT_PREFILL_LEND_IDLE` — parameter-name substrings lent but not brought
  back during a prefill, for weights the prefill never reads (GLM's hot
  experts; within 1%, so off by default). Turns `LONG_SEQ`'s first part off.
- `KT_PREFILL_LEND_PFONLY` — parameter-name substrings only the prefill
  reads: the lend in reverse, decode borrowing the prefill's memory. Such a
  weight stays out of the decode spans (it is in the pinned mirror and in the
  prefill's layer slots, nowhere else), so decode gets its VRAM back. While
  decoding it is an empty tensor: the model must not read it then. Not
  combinable with `IDLE`. A layer's span is made smaller at the end of the
  load only when it holds 1 MiB or more of those weights; the bind's slack
  alone never remakes it (a freed span stays in the pool's segment, and
  remaking every layer's ran a rank out of memory). See "The lend in reverse"
  below.
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

## The lend in reverse

The lend gives decode's weights to the prefill. Some weights are the other way
round: only the prefill reads them. On Flash-Next served from a GGUF, the
hyper-connection mix runs from int8 copies in decode (16 rows or fewer), and
its bf16 weights — 1.26 GB per card, rounded to the int8 values so both paths
compute the same thing — are read only by the prefill. With
`KT_PREFILL_LEND_PFONLY=input_mix_weight`:

- the bind leaves them out of the spans (a span that later dropped part of
  itself would keep that memory in torch_memory_saver's pool);
- after the load they are laid out at the end of their layer, go into the
  mirror and the prefill slots, the decode span ends before them, and the
  original allocations are freed;
- in a prefill they arrive with their layer like any lent weight; in decode
  they are empty tensors, and the model widens its int8 copy for the rare
  decode-phase call the int8 kernel cannot take.

Where the room goes matters. The KV pool stays resident through a prefill, the
lent weights do not; so freed decode memory is better given to lent weights
(more hot experts) than to a larger pool, which keeps that room out of every
prefill. Flash-Next (GGUF, one MTP draft step with the GGUF release's own
Q2_0 draft, two 16 GB cards, Japanese-fiction decode, 35K-token streamed
prefill at chunk 16384; two launches each, interleaved):

| | KV pool | prefill window free / peak | 35K prompt + 200 out, prompt tok/s | decode, mean of 8 |
|---|---|---|---|---|
| no reverse lend, 64 hot experts | 121,152 | 8.90 / 6.83 GB | 3,070-3,097 | 104.6-106.2 tok/s |
| reverse lend, 64 hot (pool grows) | 131,072 | 8.41 / 6.85 GB | 3,125 | 104.8 tok/s (one launch) |
| reverse lend, 84 hot, pool held | 121,152 | 8.57 / 6.85 GB | 3,043-3,097 | 107.0-109.5 tok/s |
| the same + hot experts in the KV pool's free pages | 121,152 | 8.57 / 6.85 GB | 3,070 | 110.8-111.9 tok/s |

The prompt rate there is 35,000 tokens over the whole request, the 200 output
tokens' time included. The last row is the KV lend of the next section: 18-19 more slots per layer
after the 35K, with the pool and the prefill window unchanged.

## Hot experts, and the KV pool's free pages

Two more pieces of the same serving tree, in `hot/`, for a KTransformers hybrid
whose experts come from a GGUF (`--kt-method LLAMAFILE` with a GGUF weight
path): a hot-expert cache on the GPU that follows the decode's routing, and a
lend of the KV pool's *free* pages to more hot experts.

**The cache** (`kt_hot_cache.py`, `KT_HOT_DYNAMIC=1`). The KT wrapper keeps
`--kt-num-gpu-experts` experts per layer on the GPU as raw GGUF blocks
(`kt_gguf_hot.py`, a window kernel that reads each expert once for the 2-4
tokens of a decode / MTP verify step, `kt_gguf_moe_multi.cu`). Which experts
sit there is decided from decayed route counts, a few swaps per layer every
`KT_HOT_INTERVAL` steps, copied from pinned host banks of the GGUF; the decode
graph only reads two tables (logical id -> slot, logical id -> CPU id), so the
captured graphs never change. The CPU computes exactly the routes the tables
give it (`sglang-kt-hot-cpu-mask.py` removes kt-kernel's static prefix mask),
so no route is computed twice or dropped.

**The KV lend** (`kt_hot_kv_lend.py`, `KT_HOT_KV_LEND=1`). An empty KV pool
is VRAM doing nothing. Its free pages, in groups, hold more expert slots:
lent pages still count as free to the allocator, are taken back *inside* the
allocation before any kernel runs, zeroed, and the slots they held disappear
from the tables; a streamed prefill returns everything first. The hot kernels
address every slot through per-layer pointer tables (fixed addresses, so the
graphs stay valid). The pool is never made smaller: with the lend on, the KV
pool is the same number of tokens as without it.

Flash-Next, Strata's IQ3_S GGUF (the file `Strata` serves), one-step MTP with
its Q2_0 draft:

| | hot experts per layer | decode (Japanese fiction, 1,024 tokens) | KV pool |
|---|---|---|---|
| two cards, dynamic cache, int8 hc mix | 168 | **119.9 tok/s** | 121,152 |
| two cards, identical weights to Strata (bf16 mix) | 168 | 107.0 | 121,152 |
| Strata 0.1.42 on the same two cards | | 97.2 | |
| one card + one socket, 56 fixed | 56 | 81.5-82.0 | 65,536 |
| one card + one socket, 56 fixed + the KV lend | 56 + 14-18 lent | **83.8-84.6** | 65,536 |

Paired NLL (the same 480- and 1,024-token pieces scored under each setting) is
within launch noise for the cache and identical token by token for the lend;
the cache's correctness is checked against an exact reference
(`KT_HOT_KV_LEND_CHECK=1` compares every lent slot with its bank before it is
zeroed). Two cards gain nothing from the lend (the pool's pages are the only
free VRAM at one card, not at two), so the profile has it on for one card and
off for two.

The scripts below wire the cache into another tree that already runs the KT
wrapper with GGUF experts and the lend; each takes the SGLang tree as its
argument and copies the modules from `hot/` into `python/sglang/srt/layers/moe/`:

```
python hot/sglang-kt-gguf-hot.py   <sglang tree>   # the GPU hot-expert method for GGUF experts
python hot/sglang-kt-hot-cache.py  <sglang tree>   # the cache and its tables, KT_HOT_DYNAMIC=1
python hot/sglang-kt-hot-cpu-mask.py <sglang tree> # the CPU takes the ids the cache gives it
python hot/sglang-kt-hot-kv-lend.py <sglang tree>  # the KV-page lend, KT_HOT_KV_LEND=1
```

The window kernel is built on first use with torch's `load_inline` (its source
is generated from sgl-kernel's GGUF kernels by `hot/kt-gguf-moe-multi-build.py`;
Apache-2.0 / MIT, see the file header). The cache copies experts from pinned
host banks: with the GGUF prefill streamer it takes the streamer's; without
it the cache pins its own (`KT_HOT_BANKS=own` forces this even with the
streamer), which costs the experts' size in pinned RAM, about 80 s more on
every launch, and one KV page group (65,024 instead of 65,536 tokens of pool
at one card).

Knobs: `KT_GPU_EXPERTS` (the launcher's `--kt-num-gpu-experts`),
`KT_HOT_INTERVAL` (32), `KT_HOT_SWAPS` (8 per layer per round), `KT_HOT_DECAY`
(0.9), `KT_HOT_KV_LEND_MAX` (64 lent slots per layer at most),
`KT_HOT_FUSED_ROUTE` / `KT_HOT_FUSED_COMBINE` (one Triton kernel each for the
table lookups and for the route weighting + sum; identical results), `KT_HOT_LOG`.
A seed placement (`KT_HOT_SEED`, a JSON of the hottest experts per layer from
an earlier run) shortens the warm-up; without it the cache starts from the
loader's static placement and converges within a few hundred steps.

Measured on Flash-Next. Other GGUF MoE models need their expert tensor names
(`blk.N.ffn_{gate,up,down}_exps.weight`) and a kernel for their types
(`kt_gguf_moe_multi.py` lists them).

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

With the plan (`kv-vmm/sglang-kt-kv-vmm.py`), four more: the no-op memory
saver's `region("kv_cache")` becomes `kt_kv_vmm.region()`; the scheduler
records each queued request's prompt length (`_add_request_to_queue`) and asks
`kt_prefill_plan.chunk_for` for the chunk of the batch it is about to make
(`get_new_batch_prefill`); the KV cache configurator budgets the pools under
`kt_kv_prefill_lend.swa_sizing()`, widens the SWA pool with `swa_size` and
sets up its lend with `swa_init`; the chunk-cache SWA eviction asks
`kt_prefill_plan.chunk_in_flight`. The loan itself runs from `kt_lend`'s
window: lent after the window's free memory is measured, mapped back before
the weights, and SWA pages handed back since the last forward mapped at the
next one's start.

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
