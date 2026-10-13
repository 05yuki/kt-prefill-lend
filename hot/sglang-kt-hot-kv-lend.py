"""Install kt_hot_kv_lend.py (hot experts in the KV pool's free pages,
KT_HOT_KV_LEND=1) and wire it; also copies the pieces it changed: kt_hot_cache.py
(lent slots, base tables), kt_gguf_hot.py (pointer tables in apply),
kt_gguf_moe_multi.py and kt_gguf_moe_multi/kt_gguf_moe_multi.cu (KT_PTR_TABLE
build). Default off: without KT_HOT_KV_LEND=1 every path is the one before.

- ModelRunner._init_post_memory_pool_components: register the runner's pool and
  allocator (target and MTP draft share the allocator), wrap the allocator,
  build the per-layer pointer tables before the graphs are captured;
- ModelRunner.forward, after the lend's phase switch and before the hot cache's
  policy: lend at extends / the first decode after one, give everything back at
  a streamed prefill.

Usage: python sglang-kt-hot-kv-lend.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
srt = Path(sys.argv[1]) / "python/sglang/srt"
here = Path(__file__).resolve().parent
moe = srt / "layers/moe"
for f in ("kt_hot_kv_lend.py", "kt_hot_cache.py", "kt_gguf_hot.py", "kt_gguf_moe_multi.py",
          "kt_gguf_moe_multi/kt_gguf_moe_multi.cu", "kt_gguf_moe_multi/ggml-common.h", "kt_gguf_moe_multi/vecdotq.cuh"):
    (moe / f).parent.mkdir(exist_ok=True)
    shutil.copy(here / f, moe / f)
    print(f"installed {moe / f}")


def once(path, old, new):
    s = path.read_text()
    if new in s:
        print(f"already patched {path}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"anchor found {s.count(old)} times in {path}:\n{old[:200]}")
    path.write_text(s.replace(old, new))
    print(f"patched {path}")


MR = srt / "model_executor/model_runner.py"
once(MR,
     "        that build bespoke memory pools can reuse it after allocating them.\"\"\"\n",
     "        that build bespoke memory pools can reuse it after allocating them.\"\"\"\n"
     "        # KT_HOT_KV_LEND=1 (kt_hot_kv_lend.py): hot experts in free KV pages\n"
     "        from sglang.srt.layers.moe import kt_hot_kv_lend\n"
     "\n"
     "        kt_hot_kv_lend.register_runner(self)\n")
once(MR,
     "            kt_lend.before_forward(forward_batch)\n"
     "            from sglang.srt.layers.moe import kt_hot_cache\n",
     "            kt_lend.before_forward(forward_batch)\n"
     "            from sglang.srt.layers.moe import kt_hot_kv_lend\n"
     "\n"
     "            kt_hot_kv_lend.before_forward(self, forward_batch)\n"
     "            from sglang.srt.layers.moe import kt_hot_cache\n")
