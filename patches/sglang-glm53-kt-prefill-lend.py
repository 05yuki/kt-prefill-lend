"""KT prefill lend (KT_PREFILL_LEND=1) for the GLM-5.3 tree
(source/sglang-glm53, GLM-5.3-Flash with KT NVFP4 experts): the hot experts
and the full-GPU prefill stop competing for VRAM. During a prefill of at
least --kt-gpu-prefill-token-threshold rows the GPU-resident decoder weights,
the hot experts included, give their physical memory to the prefill (the
full-GPU prefill writes all 288 experts of a layer into its own slot and never
reads the hot ones). The mechanism is patches/kt_lend.py; kt_lend recognises
this prefill by the threshold, the tree has no KT streamer.

  model_loader/loader.py     bind, after_module, after_load (this tree has an
                             NPU line after the weight processing)
  model_executor/model_runner.py
                             the phase swap before every forward; no KV-pool
                             reservation for the full-GPU slot under the lend
                             (2.11 GiB a rank; the slot comes out of the lent
                             weights instead)
  layers/moe/kt_ep_wrapper.py
                             the shared full-GPU slot (SharedFullContext, made
                             on the first qualifying prefill) in the lend
                             scratch

Usage: python sglang-glm53-kt-prefill-lend.py [sglang tree]   (kt_lend.py next to this file)
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
ROOT = Path(sys.argv[1]) / "python/sglang/srt"
LEND = ROOT / "layers/moe/kt_lend.py"
W = ROOT / "layers/moe/kt_ep_wrapper.py"
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
    # patches/sglang-glm53-kt-prefill-lend.py: before any weight loads
    from sglang.srt.layers.moe import kt_lend

    kt_lend.bind_model(model)
    return model
'''), ('''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                if _is_npu:
                    torch.npu.empty_cache()


class LayeredModelLoader(DefaultModelLoader):
''', '''                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
                if _is_npu:
                    torch.npu.empty_cache()
                # patches/sglang-glm53-kt-prefill-lend.py: a repacked layer back
                # into its span right after its module
                from sglang.srt.layers.moe import kt_lend

                kt_lend.after_module(module)

        # patches/sglang-glm53-kt-prefill-lend.py: before the KV pool is sized
        from sglang.srt.layers.moe import kt_lend

        kt_lend.after_load()


class LayeredModelLoader(DefaultModelLoader):
''')])

edit(R, "kt_lend.before_forward", [('''        ) as recorder_outputs:
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
''', '''        ) as recorder_outputs:
            # patches/sglang-glm53-kt-prefill-lend.py: weights out / scratch in
            # for a full-GPU prefill, back before anything else
            from sglang.srt.layers.moe import kt_lend

            kt_lend.before_forward(forward_batch)
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
'''), ('''                self.mxfp4_layerwise_prefill_reservation_bytes = (
                    get_full_gpu_prefill_reservation_bytes()
                )
''', '''                self.mxfp4_layerwise_prefill_reservation_bytes = (
                    get_full_gpu_prefill_reservation_bytes()
                )
                # patches/sglang-glm53-kt-prefill-lend.py: under the lend the
                # slot comes out of the lent weights, not out of the KV pool
                from sglang.srt.layers.moe import kt_lend

                if kt_lend.ENABLED:
                    self.mxfp4_layerwise_prefill_reservation_bytes = 0
''')])

edit(W, "kt_lend.scratch_region", [('''                try:
                    context = SharedFullContext(
                        layer=layer,
                        init_args=self._full_init_args,
                        global_num_experts=self.global_num_experts,
                        moe_runner_config=self.moe_runner_config,
                    )
''', '''                try:
                    # patches/sglang-glm53-kt-prefill-lend.py: the slot in the
                    # lend scratch (made inside a prefill, when it is resident)
                    from sglang.srt.layers.moe import kt_lend

                    with kt_lend.scratch_region():
                        context = SharedFullContext(
                            layer=layer,
                            init_args=self._full_init_args,
                            global_num_experts=self.global_num_experts,
                            moe_runner_config=self.moe_runner_config,
                        )
''')])
