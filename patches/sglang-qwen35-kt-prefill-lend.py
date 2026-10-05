"""KT prefill lend (KT_PREFILL_LEND=1) for the Qwen3.5 tree
(source/sglang-upstream, Qwen3.5-35B-A3B bf16 with KT experts and the bf16
streamer): during a streamed prefill the GPU-resident decoder weights, the 48
hot experts per layer included, give their physical memory to the prefill.
The mechanism is patches/kt_lend.py; the loader and model-runner points are
the same as patches/sglang-dsv41-kt-prefill-lend.py. The streamer here is its
own bf16 edition, so its edits are its own:

  - the two device slots, the zero-copy raw landings and the CUTLASS
    workspace are made in kt_lend.scratch_region() (they are made on the
    first streamed prefill, when the scratch tag is resident)
  - the graph inputs: one base per buffer (x, ids, w, acc, out) and a view of
    its head per token count, instead of a set per count; under the lend the
    bases are sized for the whole chunk at once and live in the scratch. One
    set per count in the scratch grew with every new count and ran MiMo out
    of memory at chunk 22528 (10-05).

Usage: python sglang-qwen35-kt-prefill-lend.py [sglang tree]   (kt_lend.py next to this file)
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


LEND_SRC = Path(__file__).with_name("kt_lend.py")
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
    # patches/sglang-qwen35-kt-prefill-lend.py: before any weight loads
    from sglang.srt.layers.moe import kt_lend

    kt_lend.bind_model(model)
    return model
'''), ('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-qwen35-kt-prefill-lend.py: before the KV pool is sized
        from sglang.srt.layers.moe import kt_lend

        kt_lend.after_load()


class LayeredModelLoader(DefaultModelLoader):
''')])

# a repacked layer goes back into its span right after its module, so the
# originals and the repacked copies do not pile up over the load
edit(L, "kt_lend.after_module", [('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        # patches/sglang-qwen35-kt-prefill-lend.py: before the KV pool is sized
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                from sglang.srt.layers.moe import kt_lend

                kt_lend.after_module(module)

        # patches/sglang-qwen35-kt-prefill-lend.py: before the KV pool is sized
''')])

edit(R, "kt_lend.before_forward", [('''        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
''', '''        ):
            # patches/sglang-qwen35-kt-prefill-lend.py: weights out / scratch in
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

edit(S, "kt_lend.scratch_region", [('''        self.slots = [_DeviceSlot(self.G, self.N, self.K, self.device) for _ in range(2)]
''', '''        # patches/sglang-qwen35-kt-prefill-lend.py: prefill-only buffers in
        # the lend scratch (plain allocations when lending is off)
        from sglang.srt.layers.moe import kt_lend

        with kt_lend.scratch_region():
            self.slots = [_DeviceSlot(self.G, self.N, self.K, self.device) for _ in range(2)]
'''), ('''            slot.raw = torch.empty((3, self.G, wstride), dtype=torch.uint8, device=self.device)
''', '''            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                slot.raw = torch.empty((3, self.G, wstride), dtype=torch.uint8, device=self.device)
'''), ('''            self._workspace = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
''', '''            from sglang.srt.layers.moe import kt_lend

            with kt_lend.scratch_region():
                self._workspace = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
'''), ('''        g = self._gin.get(T)
        if g is None:
            g = {
                "x": x.clone(),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
            }
            self._gin[T] = g
        else:
            g["x"].copy_(x)
''', '''        g = self._gin.get(T)
        if g is None:
            # views of one base per buffer (patches/sglang-qwen35-kt-prefill-lend.py)
            k = physical_ids.shape[1]
            g = {
                "x": self._gin_view("x", T, (T, self.K), torch.bfloat16, x.device),
                "ids": self._gin_view("ids", T, (T, k), torch.int32, x.device),
                "w": self._gin_view("w", T, (T, k), torch.float32, x.device),
                "acc": self._gin_view("acc", T, (T, self.K), torch.bfloat16, x.device),
                "out": self._gin_view("out", T, (T, self.K), torch.bfloat16, x.device),
            }
            self._gin[T] = g
        g["x"].copy_(x)
'''), ('''    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
''', '''    def _gin_view(self, name, T, shape, dtype, device):
        """The head of the base for `name`, made for max(T, 2048) rows (the
        whole chunk under the KT prefill lend, in its scratch) or grown when T
        does not fit; views of an older base keep it and its graphs alive."""
        from sglang.srt.layers.moe import kt_lend

        bases = self.__dict__.setdefault("_gin_base", {})
        esize = torch.empty((), dtype=dtype).element_size()
        need = esize
        for d in shape:
            need *= int(d)
        base = bases.get(name)
        if base is None or base.numel() < need:
            rows = max(T, 2048)
            if kt_lend.ENABLED:
                rows = max(rows, kt_lend._chunk())
            nbytes = -(-(need * rows // T) // 256) * 256
            with kt_lend.scratch_region():
                base = torch.zeros(max(nbytes, need), dtype=torch.uint8, device=device)
            bases[name] = base
        return base[:need].view(dtype).view(shape)

    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
''')])
