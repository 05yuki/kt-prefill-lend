"""Install kt_hot_cache.py (decode-adaptive hot experts, KT_HOT_DYNAMIC=1) and
wire it: the KT wrapper registers every layer after its weights are loaded
and routes decode batches through the cache's tables; model_runner.forward
runs the swap policy between steps.

Usage: python sglang-kt-hot-cache.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
srt = Path(sys.argv[1]) / "python/sglang/srt"
here = Path(__file__).resolve().parent
shutil.copy(here / "kt_hot_cache.py", srt / "layers/moe/kt_hot_cache.py")
print(f"installed {srt / 'layers/moe/kt_hot_cache.py'}")


def once(path, old, new):
    s = path.read_text()
    if new in s:
        print(f"already patched {path}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"anchor found {s.count(old)} times in {path}:\n{old[:200]}")
    path.write_text(s.replace(old, new))
    print(f"patched {path}")


W = srt / "layers/moe/kt_ep_wrapper.py"
# 1. register after the streamer is built (the banks are the swap source)
once(W,
     "            stream_prefill_init_early(self, layer)\n",
     "            stream_prefill_init_early(self, layer)\n"
     "        from sglang.srt.layers.moe import kt_hot_cache\n"
     "\n"
     "        kt_hot_cache.register(self, layer)\n")
# 2. decode routing through the cache tables
once(W,
     "        # Step 1: Submit CPU expert computation (non-blocking)\n"
     "        if self.tp_rank == 0 and streamer is None:\n"
     "            self.submit(layer, dispatch_output)\n",
     "        # KT_HOT_DYNAMIC=1 (kt_hot_cache.py): the GPU slots follow the decode's\n"
     "        # routing; the tables give the CPU the logical ids minus the resident\n"
     "        # ones and the GPU method the slot ids\n"
     "        if streamer is None and kt_hot_cache.ENABLED and kt_hot_cache._cache is not None and kt_hot_cache._cache.ready:\n"
     "            slot_ids, cpu_ids = kt_hot_cache._cache.route(self.kt_config.layer_idx, topk_output.topk_ids)\n"
     "            if self.tp_rank == 0:\n"
     "                self.submit(layer, dispatch_output._replace(topk_output=topk_output._replace(topk_ids=cpu_ids)))\n"
     "            output = self.gpu_method.apply(\n"
     "                layer, dispatch_output._replace(topk_output=topk_output._replace(topk_ids=slot_ids))\n"
     "            ).hidden_states\n"
     "            if self.tp_rank == 0:\n"
     "                output = output + self.sync(x)\n"
     "            return StandardCombineInput(hidden_states=output)\n"
     "\n"
     "        # Step 1: Submit CPU expert computation (non-blocking)\n"
     "        if self.tp_rank == 0 and streamer is None:\n"
     "            self.submit(layer, dispatch_output)\n")
s = W.read_text()
if "from sglang.srt.layers.moe import kt_hot_cache\n" not in s.split("class KTEPWrapperMethod")[0]:
    s = s.replace("from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_for\n",
                  "from sglang.srt.layers.moe import kt_hot_cache\n"
                  "from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_for\n", 1)
    W.write_text(s)
    print(f"patched {W} (import)")

# 3. the policy between steps
once(srt / "model_executor/model_runner.py",
     "            kt_lend.before_forward(forward_batch)\n",
     "            kt_lend.before_forward(forward_batch)\n"
     "            from sglang.srt.layers.moe import kt_hot_cache\n"
     "\n"
     "            kt_hot_cache.before_forward(forward_batch)\n")
