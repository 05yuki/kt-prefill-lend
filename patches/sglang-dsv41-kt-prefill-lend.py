"""KT prefill lend (KT_PREFILL_LEND=1) for the DeepSeek V4 tree
(source/sglang-dsv41: V4.1, V4-Flash, V4-Flash-Vision): during a streamed
prefill the GPU-resident decoder weights give their physical memory to the
prefill. The mechanism is patches/kt_lend.py (see its docstring and
sglang-kt-prefill-lend.py for the Flash-Next tree); this wires it in:

  model_loader/loader.py     bind after the model is built, after_load at the
                             end of load_weights_and_postprocess
  model_executor/model_runner.py
                             the phase swap before every forward
  layers/moe/kt_stream_prefill.py
                             the streamer's prefill-only buffers in the lend
                             scratch: the two DMA slots, the W16 bf16 group
                             buffer, the zero-copy raw landings, the shared
                             graph inputs, the output scratch and, when no
                             resident CUTLASS method made it first (the Marlin
                             hot-expert backend), the CUTLASS workspace

The streamer makes these on its first streamed prefill, when the scratch tag
is resident and the weights are out, so they no longer come out of what the
KV pool left. With KT_GPU_STREAM_EARLY_INIT=1 they are made at load inside
the scratch tag as well. V4's decoder is entered through
forward_hc_pre_from_prev, which kt_lend wraps (every forward* method of a
layer enters it); the engram's weights stay resident (KT_PREFILL_LEND_SKIP,
default "engram": layer 14's is prefetched on another stream).

Usage: python sglang-dsv41-kt-prefill-lend.py [sglang tree]   (kt_lend.py next to this file)
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
ROOT = Path(sys.argv[1]) / "python/sglang/srt"
LEND = ROOT / "layers/moe/kt_lend.py"
S = ROOT / "layers/moe/kt_stream_prefill.py"
R = ROOT / "model_executor/model_runner.py"
L = ROOT / "model_loader/loader.py"


def once(s, old, new, name):
    if s.count(old) != 1:
        raise SystemExit("%s: anchor found %d times:\n%s" % (name, s.count(old), old[:200]))
    return s.replace(old, new)


def edit(path, marker, pairs):
    """pairs: (old, new), or a list of (old, new) alternatives of which the
    first whose anchor is present applies (streamer variants)."""
    s = path.read_text()
    if marker in s:
        print(path.name, "already patched")
        return
    for pair in pairs:
        if isinstance(pair, list):
            pair = next((alt for alt in pair if s.count(alt[0]) == 1), pair[0])
        old, new = pair
        s = once(s, old, new, path.name)
    path.write_text(s)
    print("patched", path)


LEND_SRC = Path(__file__).resolve().parent.parent / "kt_lend.py"
if LEND.exists() and LEND.read_text() == LEND_SRC.read_text():
    print("kt_lend.py already present")
else:
    LEND.write_text(LEND_SRC.read_text())
    print("wrote", LEND)

edit(L, "kt_lend.bind_model", [('''    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    return model_class(**kwargs)
''', '''    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    model = model_class(**kwargs)
    # patches/sglang-dsv41-kt-prefill-lend.py: before any weight loads
    from sglang.srt.layers.moe import kt_lend

    kt_lend.bind_model(model)
    return model
'''), ('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-dsv41-kt-prefill-lend.py: before the KV pool is sized
        from sglang.srt.layers.moe import kt_lend

        kt_lend.after_load()


class LayeredModelLoader(DefaultModelLoader):
''')])

# a repacked layer goes back into its span right after its module, so the
# originals and the repacked copies do not pile up over the load
edit(L, "kt_lend.after_module", [('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-dsv41-kt-prefill-lend.py: before the KV pool is sized
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                from sglang.srt.layers.moe import kt_lend

                kt_lend.after_module(module)

        # patches/sglang-dsv41-kt-prefill-lend.py: before the KV pool is sized
''')])

edit(R, "kt_lend.before_forward", [('''        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
''', '''        ):
            # patches/sglang-dsv41-kt-prefill-lend.py: weights out / scratch in
            # for a streamed prefill, back before anything else
            from sglang.srt.layers.moe import kt_lend

            kt_lend.before_forward(forward_batch)
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
''')])

edit(S, "kt_lend.scratch_region", [('''        self.slots = [
            _DeviceSlot(self.G, self.N, self.K, self.device,
                        raw_dtype=None if ZEROCOPY else torch.bfloat16)
            for _ in range(2)
        ]
''', '''        # patches/sglang-dsv41-kt-prefill-lend.py: prefill-only buffers in
        # the lend scratch (plain allocations when lending is off)
        from sglang.srt.layers.moe import kt_lend

        with kt_lend.scratch_region():
            self.slots = [
                _DeviceSlot(self.G, self.N, self.K, self.device,
                            raw_dtype=None if ZEROCOPY else torch.bfloat16)
                for _ in range(2)
            ]
'''), ('''            self.w13_bf16 = torch.empty((self.G, 2 * self.N, self.K), dtype=torch.bfloat16, device=self.device)
            self.w2_bf16 = torch.empty((self.G, self.K, self.N), dtype=torch.bfloat16, device=self.device)
''', '''            with kt_lend.scratch_region():
                self.w13_bf16 = torch.empty((self.G, 2 * self.N, self.K), dtype=torch.bfloat16, device=self.device)
                self.w2_bf16 = torch.empty((self.G, self.K, self.N), dtype=torch.bfloat16, device=self.device)
'''), ('''            slot.raw = torch.empty((3, self.G, wstride), dtype=torch.uint8, device=self.device)
            slot.raw_s = torch.empty((3, self.G, sstride), dtype=torch.uint8, device=self.device)
''', '''            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                slot.raw = torch.empty((3, self.G, wstride), dtype=torch.uint8, device=self.device)
                slot.raw_s = torch.empty((3, self.G, sstride), dtype=torch.uint8, device=self.device)
'''), [('''            base = torch.zeros(nbytes, dtype=torch.uint8, device=device)
''', '''            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                base = torch.zeros(nbytes, dtype=torch.uint8, device=device)
'''), ('''        if g is None:
            g = {
                "xq": xq.clone(),
                "sf": None if sf is None else sf.clone(),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
            }
''', '''        if g is None:
            # graph inputs per token count (a streamer without the shared
            # inputs, e.g. the MiMo tree) in the lend scratch
            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                g = {
                    "xq": xq.clone(),
                    "sf": None if sf is None else sf.clone(),
                    "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                    "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                    "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                    "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
                }
''')], ('''            self._out_full = torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device)
''', '''            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                self._out_full = torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device)
'''), ('''                self._workspace = preallocate_shared_cutlass_workspace(nbytes, x.device)
''', '''                from sglang.srt.layers.moe import kt_lend

                with kt_lend.scratch_region():
                    self._workspace = preallocate_shared_cutlass_workspace(nbytes, x.device)
''')])

# shared graph-input bases sized for the chunk at once under the lend: they
# live in the scratch (resident only while prefilling), and a base grown for a
# later, longer T stays alive beside the old one (graphs hold views of both)
edit(S, "kt_lend: the whole chunk", [('''            rows = max(T, 2048)
''', '''            rows = max(T, 2048)
            from sglang.srt.layers.moe import kt_lend

            if kt_lend.ENABLED:  # kt_lend: the whole chunk
                rows = max(rows, kt_lend._chunk())
''')])
