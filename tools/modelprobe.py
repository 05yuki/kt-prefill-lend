#!/usr/bin/env python3
"""A model directory as a launch plan needs it: architecture and layer kinds from config.json; from the weights
(GGUF files, or safetensors shard headers) the bytes that have to live somewhere — experts (per layer, per expert,
by type: the RAM pool and the hot-slot size), dense (the GPU-resident rest), the token table, lm_head, PLE table,
MTP draft — and the per-token KV and per-sequence linear-attention state sizes. JSON (--json) or a table.

usage: modelprobe.py <model dir> [--gguf <dir or file>] [--json]
  <model dir>  holds config.json; its *.gguf or *.safetensors are read unless --gguf points elsewhere
Reads file headers only (no venv, no tensor data). GGUF types unknown to gguf-py (Strata's Q2_0 = 42) are fine.
"""
import argparse
import glob
import json
import os
import struct
import sys

# ggml type -> bits per weight (block bytes * 8 / block size); 42 is Strata's Q2_0 (18 B / 64)
BPW = {0: 32, 1: 16, 2: 4.5, 3: 5, 6: 5.5, 7: 6, 8: 8.5, 9: 8, 10: 2.625, 11: 3.4375, 12: 4.5, 13: 5.5, 14: 6.5625,
       15: 8, 16: 2.0625, 17: 2.3125, 18: 3.0625, 19: 1.5625, 20: 4.5, 21: 3.4375, 22: 2.5, 23: 4.25, 24: 8, 25: 16,
       26: 32, 27: 64, 28: 8, 29: 4.5, 30: 16, 42: 2.25}
TYPE_NAME = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
             12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
             20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
             29: "IQ1_M", 30: "BF16", 42: "Q2_0"}
_GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_tensors(path):
    """[(name, n_elements, ggml_type)] from one GGUF file's header."""
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        magic, version = rd("4sI")
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        n_tensors, n_kv = rd("QQ")
        skip_str = lambda: f.seek(rd("Q")[0], 1)

        def skip_value(t):
            if t in _GGUF_SCALAR:
                f.seek(struct.calcsize(_GGUF_SCALAR[t]), 1)
            elif t == 8:
                skip_str()
            elif t == 9:
                et, n = rd("IQ")
                if et in _GGUF_SCALAR:
                    f.seek(struct.calcsize(_GGUF_SCALAR[et]) * n, 1)
                else:
                    for _ in range(n):
                        skip_value(et)
            else:
                raise ValueError(f"gguf kv type {t}")

        for _ in range(n_kv):
            skip_str()
            skip_value(rd("I")[0])
        out = []
        for _ in range(n_tensors):
            name = f.read(rd("Q")[0]).decode()
            nd = rd("I")[0]
            dims = rd("Q" * nd)
            t, _off = rd("IQ")
            n = 1
            for d in dims:
                n *= d
            out.append((name, n, t))
        return out


def safetensors_tensors(path):
    """[(name, n_elements, dtype)] from one safetensors file's header."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    out = []
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        cnt = 1
        for d in info["shape"]:
            cnt *= d
        a, b = info["data_offsets"]
        out.append((name, cnt, info["dtype"], b - a))
    return out


def gguf_bytes(n, t):
    return int(n * BPW.get(t, 8) / 8)


def classify_gguf(name):
    """(bucket, layer, part) for a GGUF tensor name."""
    if name.startswith("blk."):
        _, l, rest = name.split(".", 2)
        if "_exps" in rest:
            return "experts", int(l), rest.split(".")[0]
        if rest.startswith("hc_") or rest.startswith("output_hc"):
            return "hc", int(l), rest
        return "dense", int(l), rest
    if name.startswith("mtp."):
        return "draft", None, name
    if name.startswith("per_layer_token_embd"):
        return "ple", None, name
    if name.startswith("token_embd"):
        return "embed", None, name
    if name.startswith("output.") or name == "output.weight":
        return "lm_head", None, name
    if name.startswith("output_hc"):
        return "hc", None, name
    return "dense", None, name


def classify_hf(name, num_layers, mtp_layers=0):
    m = name
    if any(k in m for k in ("visual", "vision", "image_", "audio")):
        return "vision", None, m
    if ".layers." in m:
        l = int(m.split(".layers.")[1].split(".")[0])
        if l >= num_layers:
            return ("draft" if l < num_layers + max(1, mtp_layers) else "other"), l, m
        if ".experts." in m and "shared" not in m:
            return "experts", l, m.split(".experts.")[1].split(".")[-2] if ".experts." in m else m
        if "hyper_connection" in m or ".hc_" in m:
            return "hc", l, m
        return "dense", l, m
    if "embed_tokens" in m or "token_embd" in m:
        return "embed", None, m
    if "lm_head" in m:
        return "lm_head", None, m
    if "per_layer" in m or "ple" in m.split(".")[-2:]:
        return "ple", None, m
    if "mtp" in m or "nextn" in m:
        return "draft", None, m
    return "dense", None, m


def config_summary(cfg):
    t = cfg.get("text_config", cfg)
    layer_types = t.get("layer_types")
    L = t.get("num_hidden_layers")
    if layer_types is None and t.get("full_attention_interval"):
        k = t["full_attention_interval"]
        layer_types = ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(L)]
    kinds = {}
    for lt in layer_types or []:
        kinds[lt] = kinds.get(lt, 0) + 1
    q = cfg.get("quantization_config") or t.get("quantization_config") or {}
    n_kv = t.get("num_key_value_heads") or t.get("num_attention_heads")
    head_dim = t.get("head_dim") or (t.get("hidden_size", 0) // max(1, t.get("num_attention_heads", 1)))
    full = kinds.get("full_attention", L if not layer_types else 0)
    mtp = t.get("mtp") or {}
    out = {
        "architecture": (cfg.get("architectures") or ["?"])[0], "model_type": cfg.get("model_type"),
        "num_hidden_layers": L, "layer_kinds": kinds or {"full_attention": L}, "hidden_size": t.get("hidden_size"),
        "vocab_size": t.get("vocab_size"), "max_position_embeddings": t.get("max_position_embeddings"),
        "num_experts": t.get("num_experts") or t.get("n_routed_experts"), "experts_per_tok": t.get("num_experts_per_tok"),
        "moe_intermediate_size": t.get("moe_intermediate_size"), "shared_expert_intermediate_size": t.get("shared_expert_intermediate_size"),
        "num_attention_heads": t.get("num_attention_heads"), "num_key_value_heads": n_kv, "head_dim": head_dim,
        "full_attention_layers": full, "mtp_layers": t.get("mtp_num_hidden_layers") or mtp.get("num_hidden_layers") or t.get("num_nextn_predict_layers") or 0,
        "ple": bool(t.get("ple_layer_ids")), "ple_dtype": t.get("ple_embedding_dtype"), "hc_count": t.get("hc_count"),
        "quant_method": q.get("quant_method"), "dtype": cfg.get("dtype") or cfg.get("torch_dtype"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings"),
    }
    # KV bytes per token at bf16 (the launcher's pool arithmetic; halve for fp8): MLA keeps one compressed vector
    # per token and layer (kv_lora_rank + the rope part), everything else keeps K and V per kv head
    lac = t.get("linear_attn_config") or {}
    if lac.get("full_attn_layers"):
        full = len(lac["full_attn_layers"])
        out["full_attention_layers"] = full
    if t.get("kv_lora_rank"):
        out["kv_layout"] = "mla"
        out["kv_bytes_per_token_bf16"] = full * (t["kv_lora_rank"] + (t.get("qk_rope_head_dim") or 0)) * 2
    else:
        out["kv_layout"] = "mha"
        out["kv_bytes_per_token_bf16"] = full * 2 * n_kv * head_dim * 2 if n_kv and head_dim else None
    # linear-attention state per sequence: GDN (Qwen3-Next style) or KDA (GLM-5.3 style)
    if t.get("linear_num_value_heads"):
        vh, vd, kd, conv = t["linear_num_value_heads"], t["linear_value_head_dim"], t.get("linear_key_head_dim", 0), t.get("linear_conv_kernel_dim", 4)
        lin = kinds.get("linear_attention", 0)
        out["linear_state_bytes_per_seq_bf16"] = lin * (vh * vd * kd * 2 + (vh * vd + 2 * t.get("linear_num_key_heads", 0) * kd) * (conv - 1) * 2)
    elif lac.get("kda_layers"):
        h, d, conv = lac["num_heads"], lac["head_dim"], lac.get("short_conv_kernel_size", 4)
        lin = len(lac["kda_layers"])
        out["layer_kinds"] = {"linear_attention": lin, "full_attention": full}
        out["linear_state_bytes_per_seq_bf16"] = lin * (h * d * d * 2 + 3 * h * d * (conv - 1) * 2)
    if t.get("index_topk"):
        out["indexer"] = {"topk": t["index_topk"], "head_dim": t.get("index_head_dim"), "n_heads": t.get("index_n_heads"),
                          "note": "the DSA indexer keeps its own compressed K pool per token, not counted here"}
    return out


def probe(model_dir, gguf=None):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    summ = config_summary(cfg)
    L = summ["num_hidden_layers"] or 0
    files = []
    if gguf:
        files = sorted(glob.glob(os.path.join(gguf, "*.gguf"))) if os.path.isdir(gguf) else [gguf]
    else:
        files = sorted(glob.glob(os.path.join(model_dir, "*.gguf")))
    buckets = {}  # bucket -> {"bytes", "tensors", "types": {type: bytes}}
    per_layer_experts = {}  # layer -> {part: (bytes, type)}
    fmt = None

    def add(b, nbytes, tname):
        d = buckets.setdefault(b, {"bytes": 0, "tensors": 0, "types": {}})
        d["bytes"] += nbytes
        d["tensors"] += 1
        d["types"][tname] = d["types"].get(tname, 0) + nbytes

    if files:
        fmt = "gguf"
        for f in files:
            for name, n, t in gguf_tensors(f):
                b, l, part = classify_gguf(name)
                nb = gguf_bytes(n, t)
                add(b, nb, TYPE_NAME.get(t, str(t)))
                if b == "experts":
                    per_layer_experts.setdefault(l, {})[part] = (nb, TYPE_NAME.get(t, str(t)))
    else:
        sts = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if sts:
            fmt = "safetensors"
            for f in sts:
                for name, n, dt, nb in safetensors_tensors(f):
                    b, l, part = classify_hf(name, L, summ["mtp_layers"])
                    add(b, nb, dt)
                    if b == "experts":
                        d = per_layer_experts.setdefault(l, {})
                        d[part] = (d.get(part, (0, dt))[0] + nb, dt)
    res = {"model_dir": model_dir, "weights_format": fmt, "weight_files": len(files) or len(glob.glob(os.path.join(model_dir, "*.safetensors"))),
           "config": summ, "buckets": buckets}
    E = summ["num_experts"]
    if per_layer_experts and E:
        layer_bytes = {l: sum(v[0] for v in d.values()) for l, d in per_layer_experts.items()}
        mx = max(layer_bytes.values())
        res["experts"] = {"moe_layers": len(layer_bytes), "bytes_per_layer_max": mx, "bytes_per_expert_max": mx // E,
                          "bytes_per_expert_mean": sum(layer_bytes.values()) // (E * len(layer_bytes)),
                          "bytes_total": sum(layer_bytes.values()),
                          "parts": {p: {"bytes_per_expert": v[0] // E, "type": v[1]} for p, v in per_layer_experts[max(layer_bytes, key=layer_bytes.get)].items()}}
    res["gpu_resident_bytes"] = sum(buckets.get(b, {}).get("bytes", 0) for b in ("dense", "hc", "lm_head"))
    res["draft_bytes"] = buckets.get("draft", {}).get("bytes", 0)
    res["embed_bytes"] = buckets.get("embed", {}).get("bytes", 0)
    res["ple_bytes"] = buckets.get("ple", {}).get("bytes", 0)
    return res


def table(r):
    gb = lambda b: f"{b / 2**30:.2f} GiB"
    c = r["config"]
    lines = [f"{r['model_dir']}: {c['architecture']} ({c['model_type']}), {r['weights_format']} x{r['weight_files']}, quant {c['quant_method']}, dtype {c['dtype']}",
             f"layers {c['num_hidden_layers']} {c['layer_kinds']}, hidden {c['hidden_size']}, vocab {c['vocab_size']}, context {c['max_position_embeddings']}, "
             f"mtp layers {c['mtp_layers']}, ple {c['ple']} ({c['ple_dtype']}), hc {c['hc_count']}"]
    if c["num_experts"]:
        lines.append(f"experts {c['num_experts']} top-{c['experts_per_tok']}, moe intermediate {c['moe_intermediate_size']}, shared {c['shared_expert_intermediate_size']}")
    if c.get("kv_bytes_per_token_bf16"):
        lines.append(f"KV {c['kv_bytes_per_token_bf16'] / 1024:.1f} KiB per token (bf16, {c['kv_layout']}, {c['full_attention_layers']} full-attention layers)" +
                     (f"; linear-attention state {c['linear_state_bytes_per_seq_bf16'] / 2**20:.1f} MiB per sequence" if c.get("linear_state_bytes_per_seq_bf16") else ""))
    for b, d in sorted(r["buckets"].items(), key=lambda kv: -kv[1]["bytes"]):
        types = ", ".join(f"{t} {gb(n)}" for t, n in sorted(d["types"].items(), key=lambda kv: -kv[1])[:4])
        lines.append(f"  {b:8s} {gb(d['bytes']):>11s}  {d['tensors']:5d} tensors  [{types}]")
    if "experts" in r:
        e = r["experts"]
        parts = ", ".join(f"{p} {v['type']} {v['bytes_per_expert'] / 2**20:.2f} MiB" for p, v in e["parts"].items())
        lines.append(f"expert slot (one expert, one layer): max {e['bytes_per_expert_max'] / 2**20:.2f} MiB, mean {e['bytes_per_expert_mean'] / 2**20:.2f} MiB [largest layer: {parts}]; {e['moe_layers']} MoE layers, all experts {gb(e['bytes_total'])}")
    lines.append(f"GPU-resident dense + hc + lm_head {gb(r['gpu_resident_bytes'])}; token table {gb(r['embed_bytes'])}; PLE {gb(r['ple_bytes'])}; draft {gb(r['draft_bytes'])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--gguf")
    ap.add_argument("--experts-from", help="a second checkpoint dir whose expert tensors are served (e.g. NVFP4 experts next to an fp8 dense checkpoint)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = probe(a.model_dir, a.gguf)
    if a.experts_from:
        e = probe(a.experts_from)
        r["experts"] = e.get("experts")
        r["buckets"]["experts"] = e["buckets"].get("experts", {"bytes": 0, "tensors": 0, "types": {}})
        r["experts_dir"] = a.experts_from
    print(json.dumps(r, indent=1) if a.json else table(r))


if __name__ == "__main__":
    sys.exit(main())
