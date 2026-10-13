"""Hot GPU experts from the GGUF: when the whole model comes from a GGUF
(KT_GGUF_DENSE set, --kt-method LLAMAFILE), the KT wrapper's GPU-side method
becomes kt_gguf_hot.KTGGUFHotExpertsMethod (raw GGUF blocks on the GPU,
sgl_kernel.ggml_moe_a8_vec) instead of the unquantized bf16 method the
missing quant config would give. KT_GGUF_HOT=0 keeps the old behaviour.

Installs kt_gguf_hot.py next to kt_ep_wrapper.py and patches
fused_moe_triton/layer.py.

Usage: python sglang-kt-gguf-hot.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
moe = Path(sys.argv[1]) / "python/sglang/srt/layers/moe"
here = Path(__file__).resolve().parent
shutil.copy(here / "kt_gguf_hot.py", moe / "kt_gguf_hot.py")
# the window kernel the hot method runs (kt_gguf_moe_multi.py builds it with load_inline from the .cu and headers)
shutil.copy(here / "kt_gguf_moe_multi.py", moe / "kt_gguf_moe_multi.py")
shutil.copytree(here / "kt_gguf_moe_multi", moe / "kt_gguf_moe_multi", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.bak-*"))
print(f"installed {moe / 'kt_gguf_hot.py'}")
F = moe / "fused_moe_triton/layer.py"
s = F.read_text()
old = """        if kt_config is not None:
            if self.quant_method is not None:
                gpu_method = self.quant_method
            elif quant_config is not None:
"""
new = """        if kt_config is not None:
            import os

            if (
                (os.environ.get("KT_GGUF_DENSE") or os.environ.get("KT_GGUF_HOT") == "1"
                 or (os.path.isdir(kt_config.weight_path) and any(f.endswith(".gguf") for f in os.listdir(kt_config.weight_path))))
                and kt_config.method == "LLAMAFILE"
                and os.environ.get("KT_GGUF_HOT", "1") == "1"
            ):
                # the experts are a GGUF and so is the rest of the model: the
                # hot slots hold raw GGUF blocks (kt_gguf_hot.py)
                from sglang.srt.layers.moe.kt_gguf_hot import KTGGUFHotExpertsMethod

                gpu_method = KTGGUFHotExpertsMethod(kt_config, self)
            elif self.quant_method is not None:
                gpu_method = self.quant_method
            elif quant_config is not None:
"""
if "kt_gguf_hot" in s:
    print(f"already patched {F}")
elif s.count(old) != 1:
    raise SystemExit(f"anchor found {s.count(old)} times in {F}")
else:
    F.write_text(s.replace(old, new))
    print(f"patched {F}")

F = moe / "kt_ep_wrapper.py"
s = F.read_text()
old = """        if self.num_gpu_experts > 0:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
"""
new = """        if self.num_gpu_experts > 0 and not (
            streamer is not None
            and getattr(self.gpu_method, "streams_hot_experts", False)
        ):
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
"""
if "streams_hot_experts" in s:
    print(f"already patched {F}")
elif s.count(old) != 1:
    raise SystemExit(f"anchor found {s.count(old)} times in {F} (sglang-kt-no-gpu-experts.py must be applied first)")
else:
    F.write_text(s.replace(old, new))
    print(f"patched {F}")
