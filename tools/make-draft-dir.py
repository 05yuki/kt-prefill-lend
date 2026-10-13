#!/usr/bin/env python3
"""A draft directory for Strata's MTP head (mtp-q2_0.gguf), so the GGUF serve
runs the same draft as Strata.

Strata converts the BF16 checkpoint's MTP block to its own GGUF: dense tensors
in BF16 / F32 under their checkpoint names, the routed experts in Q2_0
(mtp.layers.0.mlp.experts.gate_up_proj [512, 1280, 2560], down_proj
[512, 2560, 640]). Here the dense tensors go into model.safetensors as they
are, next to the source checkpoint's configs with the quantization config
dropped (the draft is built unquantized); the experts stay in the GGUF and are
read by kt_mtp_gguf.py (KT_MTP_GGUF, set by the launcher from kt_mtp_gguf.json).

Usage: python tools/make-draft-dir.py <mtp-q2_0.gguf> <config source dir> <out dir>
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from kt_kernel.utils.loader import GGUFLoader


def _raw_exact(loader, name):
    """Raw bytes and ggml type of a tensor by its exact name: the loader maps
    PyTorch names to llama.cpp ones, and Strata's GGUF keeps checkpoint names."""
    import kt_kernel.utils.loader as KL

    translate = KL.translate_name_to_gguf
    KL.translate_name_to_gguf = lambda n: n
    try:
        return loader.get_undequanted_tensor_and_ggml_type(name)
    finally:
        KL.translate_name_to_gguf = translate


gguf, cfg_src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
tmp = tempfile.mkdtemp()  # the loader reads a directory
os.symlink(gguf, os.path.join(tmp, gguf.name))
loader = GGUFLoader(tmp)
dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
tensors, experts = {}, []
for name, info in loader.tensor_info.items():
    kind = str(getattr(info["dtype"], "name", info["dtype"]))
    if ".mlp.experts." in name:
        experts.append(name)
        continue
    if kind not in dtypes:
        raise SystemExit(f"{name}: dense tensor of type {kind}")
    data, _ = _raw_exact(loader, name)
    shape = [int(x) for x in info["shape"]]
    tensors[name] = data.view(dtypes[kind]).reshape(shape).clone()
save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
(out / "model.safetensors.index.json").write_text(json.dumps(
    {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
     "weight_map": {k: "model.safetensors" for k in sorted(tensors)}}, indent=1))
for p in cfg_src.iterdir():
    if p.name in ("model.safetensors.index.json", "hf_quant_config.json"):
        continue
    if p.suffix in (".json", ".jinja", ".txt", ".model", ".py"):
        shutil.copy2(p, out / p.name)
cfg = json.loads((out / "config.json").read_text())
cfg.pop("quantization_config", None)
if isinstance(cfg.get("text_config"), dict):
    cfg["text_config"].pop("quantization_config", None)
(out / "config.json").write_text(json.dumps(cfg, indent=2))
(out / "kt_mtp_gguf.json").write_text(json.dumps({"gguf": str(gguf), "experts": experts}, indent=1))
gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
print(f"{len(tensors)} dense tensors ({gb:.2f} GB) -> {out}; experts left in {gguf}: {experts}")
