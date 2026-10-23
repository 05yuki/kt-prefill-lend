"""More hot experts in the KV pool's free pages outside prefill
(KT_KV_HOT_LEND=1 KT_KV_HOT_EXTRA=<X>, kt_kv_hot_lend.py): the KT wrapper's GPU
method loads N + X experts and gives X of them to pinned memory and shadow slots,
and its apply routes the lent ones to the slots.

Needs sglang-kt-kv-vmm.py on the tree first (kt_kv_vmm, kt_kv_prefill_lend) and
the prefill lend (kt_lend.py with the hot hooks).

- layers/moe/kt_ep_wrapper.py: create_weights ends with kt_kv_hot_lend.make_shadow
  (the X experts' weights on a shadow layer outside the model);
  process_weights_after_loading ends with kt_kv_hot_lend.split; apply takes
  kt_kv_hot_lend.route's two dispatches (the CPU's, the slots') and adds
  kt_kv_hot_lend.apply's output;
- layers/moe/fused_moe_triton/layer.py: the weight loader hands experts
  N..N+X-1 to the shadow (kt_kv_hot_lend.shadow_param) instead of dropping them.

Usage: python sglang-kt-kv-hot-lend.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
srt = Path(sys.argv[1]) / "python/sglang/srt"
shutil.copy(here / "kt_kv_hot_lend.py", srt / "layers/moe/kt_kv_hot_lend.py")
print(f"copied kt_kv_hot_lend.py into {srt / 'layers/moe'}")

MARK = "kt_kv_hot_lend"


def patch(rel, alternatives):
    """alternatives: lists of (old, new) pairs, the first list whose every anchor
    is found exactly once is applied."""
    f = srt / rel
    s = f.read_text()
    if MARK in s:
        print(f"already patched {f}")
        return
    for pairs in alternatives:
        if all(s.count(old) == 1 for old, _ in pairs):
            for old, new in pairs:
                s = s.replace(old, new)
            f.write_text(s)
            print(f"patched {f}")
            return
    for pairs in alternatives:
        for old, _ in pairs:
            print(f"  {s.count(old)}x {old[:80]!r}")
    raise SystemExit(f"{f}: no set of anchors matches")


def _to_shadow(indent: str) -> str:
    return (f"""{indent}    # kt_kv_hot_lend: experts N..N+X-1 load into the layer's shadow
{indent}    from sglang.srt.layers.moe import kt_kv_hot_lend

{indent}    hot = kt_kv_hot_lend.shadow_param(self.quant_method, self, param, expert_id)
{indent}    if hot is not None:
{indent}        self._weight_loader_impl(param=hot[0], loaded_weight=loaded_weight,
{indent}                                 weight_name=weight_name, shard_id=shard_id, expert_id=hot[1])
{indent}    return
""")


patch("layers/moe/fused_moe_triton/layer.py", [
    # upstream, the DeepSeek V4 trees
    [("""            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
                    return
""", """            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
""" + _to_shadow(" " * 16))],
    # the Flash-Next v0519 tree: _is_kt_cpu_expert in two loaders
    [("""            if self._is_kt_cpu_expert(expert_id):
                return
""", """            if self._is_kt_cpu_expert(expert_id):
""" + _to_shadow(" " * 12)),
     ("""        if self._is_kt_cpu_expert(expert_id):
            return
""", """        if self._is_kt_cpu_expert(expert_id):
""" + _to_shadow(" " * 8))],
])

CREATE_OLD = """        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
"""
CREATE_NEW = CREATE_OLD + """        # kt_kv_hot_lend: the next X experts' weights on a shadow layer, for the
        # KV pool's free pages
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.make_shadow(self, layer, hidden_size, intermediate_size_per_partition,
                                   params_dtype, extra_weight_attrs)
"""
SPLIT = ("""            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(""", """            self.wrapper.load_weights(physical_to_logical_map_cpu)

        # kt_kv_hot_lend: experts N..N+X-1 to pinned memory and the shadow slots
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.split(self, layer)

    def create_moe_runner(""")
CHECK = ("""        return StandardCombineInput(hidden_states=output)

    def _logical_to_physical(self, device):""", """        kt_kv_hot_lend.check(self, layer, dispatch_output, hot, output)
        return StandardCombineInput(hidden_states=output)

    def _logical_to_physical(self, device):""")
ROUTE = """
        # kt_kv_hot_lend: routes to the experts lent the KV pool's free pages go to
        # their slots, and to the CPU at weight 0
        from sglang.srt.layers.moe import kt_kv_hot_lend

        hot = kt_kv_hot_lend.route(self, dispatch_output) if streamer is None else None
"""
ADD = """        if hot is not None:
            output = output + kt_kv_hot_lend.apply(self, hot[1])
"""

patch("layers/moe/kt_ep_wrapper.py", [
    # upstream SGLang (Qwen3.5)
    [(CREATE_OLD, CREATE_NEW), SPLIT, CHECK,
     ("""        streamer = stream_prefill_for(self, layer, x.shape[0])
        if streamer is not None:
            physical_ids = topk_output.topk_ids.clone()
""", """        streamer = stream_prefill_for(self, layer, x.shape[0])
        if streamer is not None:
            physical_ids = topk_output.topk_ids.clone()
""" + ROUTE),
     ("""        if self.tp_rank == 0 and streamer is None:
            self.submit(layer, dispatch_output)
""", """        if self.tp_rank == 0 and streamer is None:
            self.submit(layer, hot[0] if hot is not None else dispatch_output)
"""),
     ("""        output = gpu_combine_input.hidden_states
        if streamer is not None:""", """        output = gpu_combine_input.hidden_states
""" + ADD + """        if streamer is not None:""")],
    # the DeepSeek V4 trees (V4.1, Vision, MiMo): resident experts optional, the
    # streamer may be built at the last layer's load
    [(CREATE_OLD, CREATE_NEW), CHECK,
     ("""            self.wrapper.load_weights(physical_to_logical_map_cpu)
        # Last layer loaded on this rank""", """            self.wrapper.load_weights(physical_to_logical_map_cpu)
        # kt_kv_hot_lend: experts N..N+X-1 to pinned memory and the shadow slots
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.split(self, layer)
        # Last layer loaded on this rank"""),
     ("""        # Step 1: Submit CPU expert computation (non-blocking)
        if streamer is None and self.tp_rank == 0:
            self.submit(layer, dispatch_output)
""", ROUTE.lstrip("\n") + """
        # Step 1: Submit CPU expert computation (non-blocking)
        if streamer is None and self.tp_rank == 0:
            self.submit(layer, hot[0] if hot is not None else dispatch_output)
"""),
     ("""        else:
            output = torch.zeros_like(x)
""", """        else:
            output = torch.zeros_like(x)
""" + ADD)],
    # the Flash-Next v0519 tree: upstream's routing, the V4 trees' load, and a GPU
    # half that may be zero on a layer
    [(CREATE_OLD, CREATE_NEW), CHECK,
     ("""            self.wrapper.load_weights(physical_to_logical_map_cpu)
        # Last layer loaded on this rank""", """            self.wrapper.load_weights(physical_to_logical_map_cpu)
        # kt_kv_hot_lend: experts N..N+X-1 to pinned memory and the shadow slots
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.split(self, layer)
        # Last layer loaded on this rank"""),
     ("""            if _routing_dump_prefix and self.tp_rank == 0:
                _routing_count(x.device, self.kt_config.layer_idx, physical_ids, self.tp_rank)
""", """            if _routing_dump_prefix and self.tp_rank == 0:
                _routing_count(x.device, self.kt_config.layer_idx, physical_ids, self.tp_rank)
""" + ROUTE),
     ("""        if self.tp_rank == 0 and streamer is None:
            self.submit(layer, dispatch_output)
""", """        if self.tp_rank == 0 and streamer is None:
            self.submit(layer, hot[0] if hot is not None else dispatch_output)
"""),
     ("""            # no GPU expert on this layer: the GPU half of the sum is zero
            output = torch.zeros_like(x)
""", """            # no GPU expert on this layer: the GPU half of the sum is zero
            output = torch.zeros_like(x)
""" + ADD)],
])


# DeepSeek V4's loader skips the checkpoint tensors of experts with no GPU slot
# (KT_LOAD_SKIP_CPU_EXPERTS): the shadow's X count as GPU experts there
f = srt / "models/deepseek_v4.py"
if f.exists() and "_kt_cpu_only_experts" in f.read_text():
    patch("models/deepseek_v4.py", [[
        ("""                if 0 <= local < module.num_local_experts and local < qm.num_gpu_experts:
""", """                # kt_kv_hot_lend: the shadow's experts load too
                if 0 <= local < module.num_local_experts and local < getattr(
                        qm, "load_gpu_experts", qm.num_gpu_experts):
"""),
    ]])


# the safetensors iterator's skip of CPU experts' tensors (KT_LOAD_SKIP_CPU_EXPERTS,
# patches/sglang-kt-load-skip-cpu-experts): the shadow's X are read too
f = srt / "model_loader/weight_utils.py"
if f.exists() and "_kt_cpu_expert_key" in f.read_text():
    patch("model_loader/weight_utils.py", [[
        ("""            n = int(moe.kt_num_gpu_experts) if moe.kt_weight_path is not None else -1
""", """            n = int(moe.kt_num_gpu_experts) if moe.kt_weight_path is not None else -1
            # kt_kv_hot_lend: the experts the hot lend's shadows hold are read too
            try:
                from sglang.srt.layers.moe import kt_kv_hot_lend

                n = kt_kv_hot_lend.load_count(n)
            except ImportError:
                pass
"""),
    ]])
