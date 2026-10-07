"""KT prefill lend, one patch (KT_PREFILL_LEND=1): during a streamed prefill
the GPU-resident weights give their physical memory to the prefill.

Folds and replaces the 10-04/05 hot-arena / dense-lend series (arena, swap,
lend, carve, partial, pool, persist, dense-lend v1/v2/fused/prefetch,
row-buckets, liblend_alloc.so). Apply on a tree with the streamer, scale-bytes
and fixed-rows patches and none of that series.

How it works (layers/moe/kt_lend.py):

  weights  Per decoder layer, every CUDA parameter of at least
           KT_PREFILL_LEND_MIN_BYTES (default 1 MiB): attention and
           linear-attention projections, hyper-connection mixes, shared
           expert, router, and the GPU-resident (hot) routed experts. Bound
           right after the model is built, before anything loads, layer by
           layer: each layer gets one contiguous span, its own allocation in
           torch_memory_saver tag "kt_lend_weights", and its parameters'
           originals are dropped before the next span is made (one arena for
           all layers at once needed the originals and 6.5 GB of arena side
           by side and ran out of memory). A tensor the loading stacks from
           several parameters and leaves them as views of (Flash-Next's
           fused qkvz+ba input projection) is found after load and moved
           into the span whole.
  scratch  What only a streamed prefill uses: the streamer's device slots and
           graph inputs, the prefill-sized CUTLASS workspace, two one-layer
           slots. Allocated in tag "kt_lend_scratch".

  Wired into the model loader, so no model file is touched: the bind after
  _initialize_model builds the model, after_load at the end of
  load_weights_and_postprocess (every module's weight processing done, the
  streamer's early init included): rebound parameters are copied back into
  the spans, fused owners moved in, the spans mirrored into pinned host
  memory, scratch paused. Then the KV pool is sized: it sees the
  decode-phase footprint. Only the first model built is bound (a draft
  model stays as it is).

  Phase swap at the model runner's forward entry, outside any graph:
    to prefill  pause weights (physical memory released, addresses kept),
                resume scratch. torch's own allocator gets the difference
                (~4 GB per rank on Flash-Next) for activations.
    to decode   empty_cache, pause scratch, resume weights, copy the mirror
                back (same addresses: the decode CUDA graphs stay valid).
  While weights are paused their addresses map nothing: a read that misses
  the per-layer hook faults at once (illegal address) instead of reading
  stale data.

  Per decoder layer, while in prefill: entering layer i (its forward, a child
  holding lent parameters, or another forward* method of the layer) points
  layer i-1 back at its span and records when its slot is free, makes layer
  i's span ready in slot i % 2 (prefetched on a side stream during layer i-1),
  points the parameters (and fused owners) at it and starts layer i+1's copy.

The streamer pads a short chunk to the next KT_GPU_STREAM_ROW_STEP (2048) rows
and runs its graphs on views of the chunk-sized inputs.

Chunk hint (KT_PREFILL_LEND_HINT=<path prefix>): the chunk has to be fixed
before the model is built, the room for it is known only after the KV pool
and the graphs. So each prefill window measures and the next launch uses it:
entering prefill records the free device memory and resets torch's peak
stats; leaving it reads the peak reserved growth and the largest forward's
rows. room = free - growth - margin (KT_PREFILL_LEND_MARGIN_MB, 512), bytes per
token = growth / rows + the chunk-proportional scratch (workspace and graph
inputs) / chunk; the hint is chunk + room / bytes per token, floored to 2048,
capped at 32768, written to <prefix>-rank<r>.json with its inputs whenever a
forward of at least 4096 rows was seen. The launcher takes the smaller of the
ranks' hints. Each rank keeps the smallest hint it has computed since it
started (the free memory at a window's start wobbles by ~50-250 MB).

Requires torch_memory_saver (hook mode "torch"; SGLang's memory saver package).

Usage: python sglang-kt-prefill-lend.py [sglang tree]   (kt_lend.py next to this file)
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
ROOT = Path(sys.argv[1]) / "python/sglang/srt"
LEND = ROOT / "layers/moe/kt_lend.py"
W = ROOT / "layers/moe/kt_ep_wrapper.py"
S = ROOT / "layers/moe/kt_stream_prefill.py"
C = ROOT / "layers/moe/moe_runner/flashinfer_cutlass.py"
Q = ROOT / "layers/quantization/modelopt_quant.py"
R = ROOT / "model_executor/model_runner.py"
L = ROOT / "model_loader/loader.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        raise SystemExit("%s: anchor found %d times:\n%s" % (name, s.count(old), old[:200]))
    return s.replace(old, new)


def edit(path, marker, pairs):
    s = path.read_text()
    if marker in s:
        print(path.name, "already patched")
        return
    for old, new in pairs:
        s = once(s, old, new, path.name)
    path.write_text(s)
    print("patched", path)


# markers only the old series and the per-model bind (6daae7b and before) write
M = ROOT / "models/qwen4_exp.py"
for p, bad in ((W, "KT_HOT_ARENA"), (S, "_kt_hot"), (C, "_hot_arena_lending"), (M, "dense_lend_bind"),
               (M, "kt_lend.bind_model"), (W, "kt_lend.after_load")):
    if p.exists() and bad in p.read_text():
        raise SystemExit(f"{p.name} carries an earlier lend ({bad}): start from the files without it")

LEND_SRC = Path(__file__).resolve().parent.parent / "kt_lend.py"
if LEND.exists() and LEND.read_text() == LEND_SRC.read_text():
    print("kt_lend.py already present")
else:
    LEND.write_text(LEND_SRC.read_text())
    print("wrote", LEND)

# -- model loader: bind right after the model is built, after_load once every
# module has processed its weights (the streamer's early init included)
edit(L, "kt_lend.bind_model", [('''    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    return model_class(**kwargs)
''', '''    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    model = model_class(**kwargs)
    # patches/sglang-kt-prefill-lend.py: before any weight loads
    from sglang.srt.layers.moe import kt_lend

    kt_lend.bind_model(model)
    return model
'''), ('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-kt-prefill-lend.py: before the KV pool is sized
        from sglang.srt.layers.moe import kt_lend

        kt_lend.after_load()


class LayeredModelLoader(DefaultModelLoader):
''')])

# -- model runner: the phase swap -------------------------------------------
# a repacked layer goes back into its span right after its module, so the
# originals and the repacked copies do not pile up over the load
edit(L, "kt_lend.after_module", [('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-kt-prefill-lend.py: before the KV pool is sized
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                from sglang.srt.layers.moe import kt_lend

                kt_lend.after_module(module)

        # patches/sglang-kt-prefill-lend.py: before the KV pool is sized
''')])

# Flash-Next stacks in_proj_qkvz and in_proj_ba into one weight per layer at
# the end of load_weights, after the originals moved into the spans: the
# copies of all 48 layers sat on top of the spans at once and one card (TP=1)
# ran out of memory there. Each layer goes back into its span right after.
edit(M, "kt_lend.after_module(module)  # fused in_proj", [('''        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()
''', '''        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()
                from sglang.srt.layers.moe import kt_lend

                kt_lend.after_module(module)  # fused in_proj
''')])

edit(R, "kt_lend.before_forward", [('''        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
''', '''        ):
            # patches/sglang-kt-prefill-lend.py: weights out / scratch in for a
            # streamed prefill, back before anything else
            from sglang.srt.layers.moe import kt_lend

            kt_lend.before_forward(forward_batch)
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
''')])

# -- CUTLASS workspace: decode-sized outside a prefill, chunk-sized in scratch
edit(C, "_lent_cutlass_workspace", [('''def get_shared_cutlass_workspace(nbytes: int, device: torch.device) -> Optional[torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    ws = _shared_cutlass_workspace.get(key)
''', '''# patches/sglang-kt-prefill-lend.py: the prefill-sized workspace lives in
# the lend scratch and is handed out only while a streamed prefill runs; the
# one above only covers the calls outside it (decode graphs bake it in).
_lent_cutlass_workspace = {}


def _kt_lend_phase():
    from sglang.srt.layers.moe import kt_lend

    return kt_lend.ENABLED, kt_lend.phase()


def get_shared_cutlass_workspace(nbytes: int, device: torch.device) -> Optional[torch.Tensor]:
    key = device.index if device.index is not None else torch.cuda.current_device()
    lend_on, phase = _kt_lend_phase()
    if lend_on and phase == "prefill" and key in _lent_cutlass_workspace:
        ws = _lent_cutlass_workspace[key]
        if ws.numel() >= nbytes:
            return ws
        _logger.warning("lent CUTLASS workspace too small (%d MB needed, %d MB)",
                        nbytes >> 20, ws.numel() >> 20)
        return None
    ws = _shared_cutlass_workspace.get(key)
'''), ('''def _workspace_max_tokens(num_tokens: int) -> int:
    try:
''', '''def _workspace_max_tokens(num_tokens: int) -> int:
    lend_on, phase = _kt_lend_phase()
    if lend_on and phase != "prefill":
        return num_tokens
    try:
''')])

edit(Q, "kt_lend.scratch_empty", [('''            w2 = layer.w2_weight  # [E_local, K, N/2] packed nibbles
            experts = max(w2.shape[0], int(os.environ.get("KT_GPU_STREAM_GROUP", "0") or 0))
            nbytes = fp4_workspace_nbytes(
                1, w2.shape[1], w2.shape[2] * 2, experts * layer.moe_ep_size, layer.top_k,
                _activation_type(moe_runner_config), layer.moe_tp_size, layer.moe_tp_rank,
                layer.moe_ep_size, layer.moe_ep_rank, w2.device)
            preallocate_shared_cutlass_workspace(nbytes, w2.device)
''', '''            w2 = layer.w2_weight  # [E_local, K, N/2] packed nibbles
            experts = max(w2.shape[0], int(os.environ.get("KT_GPU_STREAM_GROUP", "0") or 0))
            from sglang.srt.layers.moe import kt_lend

            if kt_lend.ENABLED:
                # patches/sglang-kt-prefill-lend.py: the decode graphs get a
                # workspace for the calls below the streaming threshold; the
                # chunk-sized one lives in the lend scratch
                from sglang.srt.layers.moe.kt_stream_prefill import STREAM_THRESHOLD
                from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as _fc
                from sglang.srt.server_args import get_global_server_args

                key = w2.device.index
                if w2.shape[0]:  # no resident experts (hot 0): no call below the threshold
                    small = fp4_workspace_nbytes(
                        max(STREAM_THRESHOLD, 1), w2.shape[1], w2.shape[2] * 2, w2.shape[0] * layer.moe_ep_size,
                        layer.top_k, _activation_type(moe_runner_config), layer.moe_tp_size, layer.moe_tp_rank,
                        layer.moe_ep_size, layer.moe_ep_rank, w2.device)
                    preallocate_shared_cutlass_workspace(small, w2.device)
                if key not in _fc._lent_cutlass_workspace:
                    big = fp4_workspace_nbytes(
                        get_global_server_args().chunked_prefill_size, w2.shape[1], w2.shape[2] * 2,
                        experts * layer.moe_ep_size, layer.top_k, _activation_type(moe_runner_config),
                        layer.moe_tp_size, layer.moe_tp_rank, layer.moe_ep_size, layer.moe_ep_rank, w2.device)
                    _fc._lent_cutlass_workspace[key] = kt_lend.scratch_empty((big,), torch.uint8, w2.device,
                                                                             per_chunk=True)
            else:
                nbytes = fp4_workspace_nbytes(
                    1, w2.shape[1], w2.shape[2] * 2, experts * layer.moe_ep_size, layer.top_k,
                    _activation_type(moe_runner_config), layer.moe_tp_size, layer.moe_tp_rank,
                    layer.moe_ep_size, layer.moe_ep_rank, w2.device)
                preallocate_shared_cutlass_workspace(nbytes, w2.device)
''')])

# -- streamer: slots and chunk-sized graph inputs in scratch, row steps -----
edit(S, "_prealloc_gin", [('''class _DeviceSlot:
    def __init__(self, G: int, N: int, K: int, device):
        u8 = torch.uint8
        self.w13_weight = torch.empty((G, 2 * N, K // 2), dtype=u8, device=device)
        self.w2_weight = torch.empty((G, K, N // 2), dtype=u8, device=device)
        self.w13_scale = torch.empty((G, 2 * N, K // 16), dtype=u8, device=device)
        self.w2_scale = torch.empty((G, K, N // 16), dtype=u8, device=device)
''', '''class _DeviceSlot:
    def __init__(self, G: int, N: int, K: int, device):
        from sglang.srt.layers.moe import kt_lend

        u8 = torch.uint8

        def empty(shape):
            # patches/sglang-kt-prefill-lend.py: resident only during a prefill
            return kt_lend.scratch_empty(shape, u8, device)

        self.w13_weight = empty((G, 2 * N, K // 2))
        self.w2_weight = empty((G, K, N // 2))
        self.w13_scale = empty((G, 2 * N, K // 16))
        self.w2_scale = empty((G, K, N // 16))
'''), ('''        self.raw = torch.empty((3, G, _align64(N * K // 2)), dtype=u8, device=device)
        self.raw_s = torch.empty((3, G, _align64(N * (K // 16) * 4)), dtype=u8, device=device)
''', '''        self.raw = empty((3, G, _align64(N * K // 2)))
        self.raw_s = empty((3, G, _align64(N * (K // 16) * 4)))
'''), ('''_prealloc = {}
''', '''_prealloc = {}
# device index -> (chunk rows, graph inputs sized for it, in the lend scratch)
_prealloc_gin = {}
# Pad short chunks to the next multiple of this many rows, not to the chunk.
ROW_STEP = int(os.environ.get("KT_GPU_STREAM_ROW_STEP", "2048") or 0)
'''), ('''    slots = [_DeviceSlot(GROUP_SIZE, N, K, device) for _ in range(2)]
    _prealloc[key] = slots
''', '''    slots = [_DeviceSlot(GROUP_SIZE, N, K, device) for _ in range(2)]
    _prealloc[key] = slots
    # the graph inputs for the one chunk length the streamer runs (shorter
    # steps use views of their head), made now so no long-lived buffer is
    # allocated inside a prefill
    if FIXED_ROWS and _chunk_rows() > 0:
        from sglang.srt.layers.moe import kt_lend

        T, topk, f32 = _chunk_rows(), layer.top_k, torch.float32

        def buf(shape, dtype):
            return kt_lend.scratch_empty(shape, dtype, device, per_chunk=len(shape) > 0 and shape[0] == T)

        _prealloc_gin[key] = (T, {
            "x": buf((T, K), torch.bfloat16), "ids": buf((T, topk), torch.int32),
            "w": buf((T, topk), f32), "acc": buf((T, K), torch.bfloat16),
            "out": buf((T, K), torch.bfloat16), "a1": buf((), f32), "a2": buf((), f32),
            "s2g": buf((GROUP_SIZE,), f32), "s2d": buf((GROUP_SIZE,), f32),
            "g1": buf((GROUP_SIZE,), f32), "g2": buf((GROUP_SIZE,), f32),
        })
'''), ('''        g = self._gin.get(T)
        n = x.shape[0]
        if g is None:
            f32 = torch.float32
''', '''        g = self._gin.get(T)
        n = x.shape[0]
        pre = _prealloc_gin.get(self.device.index)
        if g is None and pre is not None and pre[0] >= T:
            g = {k: (v[:T] if k in ("x", "ids", "w", "acc", "out") else v) for k, v in pre[1].items()}
            g["x"].zero_()
            g["acc"].zero_()
            for k in ("a1", "a2", "s2g", "s2d", "g1", "g2"):
                g[k].fill_(1.0)
            self._gin[T] = g
        if g is None:
            f32 = torch.float32
'''), ('''        rows = x.shape[0]
        T = rows
        if FIXED_ROWS and rows < _chunk_rows():
            T = _chunk_rows()
''', '''        rows = x.shape[0]
        T = rows
        if FIXED_ROWS and rows < _chunk_rows():
            T = _chunk_rows()
            if ROW_STEP > 0:
                T = min(T, -(-rows // ROW_STEP) * ROW_STEP)
''')])
