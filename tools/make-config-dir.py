#!/usr/bin/env python3
"""A config directory for serving a model from its GGUF: the checkpoint's config.json and tokenizer files, with
the quantization config set to gguf (the dense layers are served in their GGUF types) and the PLE table's dtype
taken from the GGUF, so SGLang loads no safetensors at all (KT_GGUF_DENSE points at the GGUF directory).

usage: make-config-dir.py <hf dir with config.json + tokenizer> <gguf dir> <out dir> [--keep-quant]
  --keep-quant   keep modules_to_not_convert as in the HF config instead of the Flash-Next list below
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modelprobe import TYPE_NAME, gguf_tensors  # noqa: E402

# Flash-Next: the dense modules that stay unquantized under the GGUF quant method (PLE, hyper-connection mixes,
# indexer, GDN alpha / beta, the router gates, conv1d, the MTP draft)
FLASH_NEXT_NOT_CONVERT = ["ple", "hyper_connection", "indexer", "in_proj_a", "in_proj_b", "mlp.gate",
                          "shared_expert_gate", "conv1d", "mtp"]
COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "added_tokens.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir")
    ap.add_argument("gguf_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--keep-quant", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.hf_dir, "config.json")))
    ple_type = None
    for f in sorted(glob.glob(os.path.join(a.gguf_dir, "*.gguf"))):
        for name, _n, t in gguf_tensors(f):
            if name.startswith("per_layer_token_embd"):
                ple_type = TYPE_NAME.get(t, str(t)).lower()
    qc = {"quant_method": "gguf"}
    if a.keep_quant and isinstance(cfg.get("quantization_config"), dict):
        qc["modules_to_not_convert"] = cfg["quantization_config"].get("modules_to_not_convert", [])
    else:
        qc["modules_to_not_convert"] = FLASH_NEXT_NOT_CONVERT
    cfg["quantization_config"] = qc
    t = cfg.get("text_config", cfg)
    if ple_type:
        t["ple_embedding_dtype"] = ple_type
    cfg["_gguf_dense_note"] = ("config/tokenizer only; weights come from the GGUF named by KT_GGUF_DENSE (dense + PLE) "
                               "and --kt-weight-path (experts)")
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    copied = []
    for n in COPY:
        p = os.path.join(a.hf_dir, n)
        if os.path.exists(p):
            shutil.copy(p, a.out_dir)
            copied.append(n)
    print(f"{a.out_dir}: config.json (quant gguf, ple {ple_type}), copied {', '.join(copied)}")


if __name__ == "__main__":
    sys.exit(main())
