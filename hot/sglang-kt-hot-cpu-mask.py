"""The CPU experts take every id the hot cache gives them (KT_HOT_DYNAMIC=1).

The KT wrapper built kt-kernel's MoE with a prefix gpu_experts_mask: logical ids
0..num_gpu_experts-1 marked as GPU experts, which kt-kernel skips
(`MOEConfig::should_skip_expert`, as it skips -1). That is right for a static
placement, where the GPU slots are physical 0..N-1 and the GPU method computes
exactly those. Under the dynamic hot cache the slots hold the experts the cache
chose (seeded from a map, then swapped), and residency is already carried by the
ids: the CPU gets the logical ids with the resident ones set to -1. So every
route to a non-resident expert with a logical id below N was computed by
nobody — neither the slots (not resident) nor the CPU (masked). On the 10-12
capture (TP2 default, hot 104, 200-token extend pieces) 5-9% of a layer's routes
were dropped; the CPU output recomputed offline with the prefix mask is
bit-identical to the served one (tools/hot-numerics/served_split.py). Decode
dropped such routes too (forward_one checks the same mask).

With the dynamic cache the mask is now empty. KT_HOT_CPU_PREFIX_MASK=1 restores
the old mask (comparison only).

Usage: python sglang-kt-hot-cpu-mask.py <sglang tree>
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
W = Path(sys.argv[1]) / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"

OLD = ("            gpu_experts_mask = _prefix_gpu_experts_mask(\n"
       "                num_experts, self.num_gpu_experts\n"
       "            )\n")
NEW = ("            # KT_HOT_DYNAMIC=1 (kt_hot_cache.py): the slots hold whatever the\n"
       "            # cache put there and the CPU gets the logical ids with the\n"
       "            # residents set to -1, so kt-kernel must not skip ids on its own.\n"
       "            # The prefix mask (ids 0..N-1 = GPU) is right only when the GPU\n"
       "            # slots are 0..N-1; under the dynamic cache it dropped every\n"
       "            # non-resident route to an expert below N (sglang-kt-hot-cpu-mask.py).\n"
       "            # KT_HOT_CPU_PREFIX_MASK=1 restores it (comparison only).\n"
       "            gpu_experts_mask = _prefix_gpu_experts_mask(\n"
       "                num_experts,\n"
       "                0 if (kt_hot_cache.ENABLED and os.environ.get(\"KT_HOT_CPU_PREFIX_MASK\", \"0\") != \"1\")\n"
       "                else self.num_gpu_experts,\n"
       "            )\n")

s = W.read_text()
if NEW in s:
    print(f"already patched {W}")
elif s.count(OLD) != 1:
    raise SystemExit(f"anchor found {s.count(OLD)} times in {W}")
else:
    if "from sglang.srt.layers.moe import kt_hot_cache" not in s or "\nimport os\n" not in s:
        raise SystemExit(f"{W}: expected `import os` and the kt_hot_cache import at the top")
    W.write_text(s.replace(OLD, NEW))
    print(f"patched {W}")
