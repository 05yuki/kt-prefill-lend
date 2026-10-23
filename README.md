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

Five more models, including DeepSeek V4.1 at a 1M-token context, are in
[Results](#results). The details behind every section are in
[docs/details.md](docs/details.md).

## How it works

In a KTransformers hybrid (CPU experts; attention and hot experts on the GPU),
a long prompt is prefilled by streaming the experts over PCIe, chunk by chunk.
Every forward has a fixed transfer cost, so a larger chunk is faster, but on a
16 GB card the chunk only gets what decode leaves free.

During a prefill the lend releases the decoder weights' VRAM with
[torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) (the
addresses stay, so the decode CUDA graphs stay valid) and streams each layer
back from a pinned host copy just before it runs. Everything is back before
the next decode step. The chunk grows, the forward count drops, and the KV
pool keeps every token it would have without the lend.

Optionally (**the plan**), a prefill also borrows the KV pool's free pages, and
the chunk is sized per request from the memory free at that moment.

## Results

2x RTX 5070 Ti 16 GB (PCIe 4.0 x16, no P2P), dual EPYC 7452, CPU experts in
kt-kernel (AVX2). Tokens/s, without -> with the lend, on the same server.

| Model | chunk | 38K prompt | 114K prompt | decode | quality |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | 2048 -> 16384 | 1,336-1,337 -> **4,153-4,154** | 1,367 -> **4,596** (105K) | 58.5-58.7 -> 54.8-59.3 | bit-identical |
| Qwen3.5-35B-A3B bf16 | 2048 -> 32768 | 1,568 -> **6,492-6,507** | 1,533 -> **6,406** | 91-97 -> 98-100 | bit-identical |
| MiMo-V2.6-Flash MXFP4 | 2048 -> 22528 | 525 -> **1,962-1,978** | 496 -> **1,550** | 30-32 -> 32 | within run-to-run spread |
| DeepSeek V4.1-Flash MXFP4, 1M ctx | 2048 -> 6144 | 255 -> **576** | 259 -> **596** | 25.4-25.6 -> 27.6-28.0 | NLL z -1.90 |
| DeepSeek V4-Flash-Vision MXFP4 | 2048 -> 16384 | 601 -> **900-919** | 598 -> **849** | 36.5-39.9 -> 42.5-44.8 | NLL z -0.58 |
| GLM-5.3-Flash NVFP4 experts | 2048 -> 14336 | 163 -> **327** | 155 -> **292** | 18.3 -> 19.9-20.1 | NLL z -0.35 / -1.04 |

TP=2 over both sockets. V4.1 keeps its 1,048,576-token KV pool and prefills a
989,154-token prompt at 406 tok/s. On one card and one socket the lend is what
makes room for a prefill at all: Qwen3.5 893-894 -> **5,142-5,157** tok/s at
38K, Flash-Next **3,030-3,069** where it ran out of memory without it. Vision
and GLM also gained hot experts (10 -> 12, 0 -> 10).

*Bit-identical*: top-20 next-token logprobs match the base server bit for bit.
*NLL z*: the same tokens read teacher-forced on both servers, z of the mean
log-prob difference (negative = lend side lower).

**With the plan**, a 35,000-token prompt (tokens/s, without -> with the plan):

| Model | into an empty pool | with another 35K cached | KV pool |
|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | 4,736 -> **5,452** | 4,834 -> 4,808 | same |
| Qwen3.5-35B-A3B bf16 | 6,446 -> **8,178** | 6,809 -> **8,883** | same |
| DeepSeek V4.1-Flash, 1M ctx | 452 -> **603** | 609 -> **787** | 1,027,072 -> 1,048,576 |
| DeepSeek V4-Flash-Vision | | 839 -> 857 | same |
| MiMo-V2.6-Flash | | 1,915 -> 1,834 | same |
| GLM-5.3-Flash NVFP4 experts | 320 -> 321 | 322 -> 323 | 201,280 -> **217,408** |
| Qwen3.8-Flash-Next GGUF (IQ3_S) | | 4,636 -> 4,391 | same |

It gains where the chunk count drops; Vision, MiMo and GLM keep theirs at 35K.
Decode is untouched, NLL identical or within launch noise, no run out of
memory. With prompts cached around every lending window, hashing every page in
use before and after found 0 changed (Flash-Next, MiMo, Vision). Empty cells:
the first long prompt of that launch carried the streamer's one-time setup.

## Quick start

**From nothing** (Flash-Next from a GGUF: both patched trees, a venv, kt-kernel
built for your CPU and GPU, a sized launch):

```sh
tools/install.sh ~/kt-serve                                                 # 10-20 min
tools/make-config-dir.py <HF dir> <gguf dir> ~/kt-serve/config
~/kt-serve/venv/bin/python tools/make-draft-dir.py <mtp gguf> <HF dir> ~/kt-serve/draft   # optional
tools/serve.py --profile flash-next-gguf --venv ~/kt-serve/venv --tree ~/kt-serve/sglang \
               --config-dir ~/kt-serve/config --gguf <gguf dir> --draft-dir ~/kt-serve/draft --tp 1 --calibrate
```

`serve.py` probes the host and the model and prints the command with its
arithmetic (`--dry-run` stops there). The context is the model's own and the
KV pool is the context; the card's remainder goes to hot experts, and the
memory fraction follows from that. `--calibrate` reads the first launch's log,
keeps what it measured and relaunches if the plan moved (later launches start
from those numbers). The plan is on when the tree has kv-vmm built
(`install.sh` builds it; `--no-plan` turns it off). Profiles:
`flash-next-gguf`, `flash-next-nvfp4`, `glm53-flash`, `dsv4-flash-vision`,
`mimo-v26-flash`.

**On a server you already run:**

```sh
pip install torch_memory_saver==0.0.9.post1          # in the server's venv
python patches/sglang-kt-stream-graph-keep.py <sglang tree>
python patches/sglang-kt-stream-shared-graph-inputs.py <sglang tree>   # DeepSeek V4 trees only
python patches/sglang-kt-prefill-lend.py <sglang tree>                 # the patch for your tree, below
KT_PREFILL_LEND=1 <your launcher>                                      # first launch calibrates, ~2 min
```

| Tree | Patch |
|---|---|
| Qwen3.8-Flash-Next (SGLang v0519 fork) | `patches/sglang-kt-prefill-lend.py` |
| DeepSeek V4 and its copies (V4.1, Vision, MiMo) | `patches/sglang-dsv41-kt-prefill-lend.py` |
| Qwen3.5 (upstream SGLang, bf16 streamer) | `patches/sglang-qwen35-kt-prefill-lend.py` |
| GLM-5.3 (full-GPU prefill) | `patches/sglang-glm53-kt-prefill-lend.py` |

Also patch the installed kt-kernel (`patches/kt-kernel-shared-gpu-output.py`,
then `patches/kt-kernel-lend-gpu-output.py`, on `kt_kernel/experts_base.py`),
and source `tools/kt-lend-auto.sh` in your launcher to have the chunk picked
for you. For the plan: `kv-vmm/build.sh`, then
`python kv-vmm/sglang-kt-kv-vmm.py <sglang tree>` and the row-block patch for
your streamer; for the hot lend, `python kv-vmm/sglang-kt-kv-hot-lend.py <sglang tree>`
and `KT_KV_HOT_LEND=1 KT_KV_HOT_EXTRA=<X>`. Every step in full:
[docs/details.md](docs/details.md#applying-it).

Then send a long prompt and watch `nvidia-smi`: usage climbs for the prefill
and drops back for decode. `KT_PREFILL_LEND=0` turns it off.

## Hot experts (GGUF)

`hot/` adds a GPU hot-expert cache that follows decode's routing, and lends
the KV pool's free pages to more hot experts in decode. Flash-Next from
Strata's IQ3_S GGUF, one MTP draft step, decode:

| | hot experts per layer | decode | KV pool |
|---|---|---|---|
| two cards, dynamic cache | 168 | **119.9 tok/s** | 121,152 |
| Strata 0.1.42 on the same two cards | | 97.2 | |
| one card + one socket | 56 | 81.5-82.0 | 65,536 |
| one card + one socket + the KV lend | 56 + 14-18 lent | **83.8-84.6** | 65,536 |

## Hot experts in the KV pool's free pages (NVFP4, MXFP4)

`kv-vmm/kt_kv_hot_lend.py` does the same for the GPU's own expert method:
outside prefill the plan's free KV pages hold more hot experts, and a prefill
or a request that needs the pages takes them back first. Decode, TP=2:

| Model | slots lent | decode without -> with |
|---|---|---|
| Qwen3.8-Flash-Next NVFP4 | X = 32 asked | 65.57 / 66.40 -> **67.97 / 67.69** |
| DeepSeek V4.1-Flash MXFP4, 1M pool | 3-4 a layer | 28.95 -> **29.98** |
| MiMo-V2.6-Flash MXFP4 | 3-4 a layer | 31.97 -> **32.80** |
| DeepSeek V4-Flash-Vision MXFP4 | 2-3 a layer | 41.90 -> **42.57 / 43.54** |

It pays where the CPU experts set the token time; a GPU-bound model (Qwen3.5
bf16) loses a little. Setup and the NLL notes:
[docs/details.md](docs/details.md#hot-experts-and-the-kv-pools-free-pages).

## Requirements and limits

- Who it is for: people running a KTransformers hybrid who are comfortable
  patching an SGLang tree. The expert streamer the patches edit is not in
  upstream SGLang (`streamers/` has the versions they expect).
- A streamed prefill (or GLM's full-GPU prefill); prefill runs eagerly, decode
  graphs are fine.
- Tested on one host: TP=2 over two sockets, TP=2 on one socket, TP=1 on one
  card and one socket. Not tested: EP, DP attention, PP.
- `expandable_segments` is refused by torch_memory_saver.
- A model that reads another layer's weights during prefill faults instead of
  reading stale data; `KT_PREFILL_LEND_SKIP` / `KT_PREFILL_LEND_RESIDENT` keep
  such weights resident.
- Measured one request at a time; concurrent serving is not measured beyond
  the window transitions waiting for in-flight GPU work.

The rest, and every knob: [docs/details.md](docs/details.md).

## Related work

[deepseek-v4.1-flash-1017-ktransformers](https://github.com/05yuki/deepseek-v4.1-flash-1017-ktransformers)
runs all of this on SGLang main for DeepSeek V4.1-Flash-1017: the lend, the
plan, a hot lend whose slots follow the routing, and the KV pool past the GPU
in host memory (kv-vmm's split ranges), so one 16 GB card serves a 1M context.

[Strata](https://github.com/Niko1221/Strata) lends slots of its GPU expert
cache to the prompt path and refills them afterwards, inside its own engine.
This patch lends the dense decoder weights as well, inside SGLang, keeping the
decode graphs' addresses through torch_memory_saver.

## License

Apache-2.0, like KTransformers and SGLang.
