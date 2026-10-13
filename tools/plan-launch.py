#!/usr/bin/env python3
"""host (hostprobe) + model (modelprobe) + intent -> the launch knobs, with the arithmetic shown.

usage: plan-launch.py <model dir> [--gguf DIR] [--tp N] [--context TOKENS] [--kv-dtype bf16|fp8] [--mamba-slots N]
                      [--mem-fraction F] [--embed-host 0|1] [--draft-bytes B] [--host-json FILE] [--model-json FILE]
                      [--env]
Per rank, the VRAM budget is  vram_total * mem_fraction  minus the CUDA context; from it come, in this order:
  dense weights (sharded by TP; the hyper-connection mixes and norms are replicated), lm_head (sharded), the token
  table (0 if it lives in pinned host memory), the MTP draft, the KV pool for --context tokens (never reduced: the
  pool is the context), the linear-attention states for --mamba-slots sequences, the CUDA graphs and the attention
  workspace, and what is left is hot expert slots (the slot is the model's mean expert size). Threads for the CPU
  expert pools are each node's physical cores minus two; the prefill chunk is what the free VRAM after the pool
  allows for the streamed prefill (~365 KB a token with the chunk-sized scratch, measured 10-13). The constants
  (CUDA context, graphs, workspace, per-token scratch) are the measured ones of the Flash-Next GGUF stack and are
  printed, so a different model can be corrected from its own launch log. --env prints the launcher's env line.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GiB = 2**30
# measured on the 5070 Ti / Flash-Next GGUF launches (native-ubuntu/GGUF-EXPERTS-20261007.md, 10-13 logs)
CUDA_CONTEXT = 0.50 * GiB      # driver + context per process
GRAPHS = 0.35 * GiB            # target verify + draft extend graphs
WORKSPACE = 1.00 * GiB         # attention / sampling workspace kept after the pool is sized (decode-vram frees ~0.9)
PREFILL_SCRATCH_PER_TOKEN = 365 * 1024   # streamed prefill: scratch per chunk token
CHUNK_MAX = 16384


def load(cmd, path):
    if path:
        return json.load(open(path))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr)
    return json.loads(out.stdout)


def plan(host, model, a):
    c = model["config"]
    tp = a.tp
    gpus = sorted(host["gpus"], key=lambda g: g["numa_node"])
    if len(gpus) < tp:
        raise SystemExit(f"{len(gpus)} GPUs for tp {tp}")
    ranks = gpus[:tp]
    vram = min(g["vram_total"] for g in ranks)
    notes = []
    b = model["buckets"]
    dense = b.get("dense", {}).get("bytes", 0) / tp + b.get("hc", {}).get("bytes", 0)  # hc replicated
    lm_head = b.get("lm_head", {}).get("bytes", 0) / tp
    if c.get("tie_word_embeddings") and not lm_head and a.embed_host:
        lm_head = model["embed_bytes"] / tp  # tied, table in host memory: a GPU copy still serves as lm_head
    # tied with the table on the GPU: one tensor, already counted as the token table
    embed = 0 if a.embed_host else model["embed_bytes"] / tp
    draft = a.draft_bytes if a.draft_bytes is not None else (model["draft_bytes"] / tp or (0.84 * GiB if c["mtp_layers"] else 0))
    if a.draft_bytes is None and model["draft_bytes"] == 0 and c["mtp_layers"]:
        notes.append("draft bytes not in this checkpoint: 0.84 GiB assumed (the Flash-Next draft dir, Q2_0 experts); --draft-bytes 0 if no draft is served")
    # MLA keeps the whole compressed KV on every rank; head-sharded KV splits by tp
    kv_tok = (c.get("kv_bytes_per_token_bf16") or 0) / (1 if c.get("kv_layout") == "mla" else tp)
    if a.kv_dtype == "fp8":
        kv_tok /= 2
    kv = a.context * kv_tok + (c.get("kv_window_bytes_bf16") or 0) / tp * (0.5 if a.kv_dtype == "fp8" else 1)
    lin = (c.get("linear_state_bytes_per_seq_bf16") or 0) / tp * a.mamba_slots
    cal = a.constants_dict
    if cal:
        # constants calibrated from a launch log of this profile (tools/plan-calibrate.py): the measured fixed
        # weights (dense, hc, lm_head, token table, draft as that launch had them), the real bytes a token takes
        # (index pools included), the states, and what the server keeps beside the pool
        kv_scale = 1.0 if cal.get("kv_dtype", "bf16") == a.kv_dtype else (0.5 if a.kv_dtype == "fp8" else 2.0)
        kv_tok = cal["kv_bytes_per_token"] * kv_scale
        kv = a.context * kv_tok
        lin = cal["states_bytes"] / max(1, cal.get("states_slots") or 1) * a.mamba_slots
        budget = vram * a.mem_fraction
        fixed = cal["weights_fixed"] + kv + lin + cal["overhead"] + (0 if cal.get("includes_draft") else draft)
        notes.append(f"constants from {os.path.basename(os.path.dirname(cal['from']['log']))}: fixed weights "
                     f"{cal['weights_fixed'] / GiB:.2f} GiB, {kv_tok / 1024:.2f} KiB a token, overhead {cal['overhead'] / GiB:.2f} GiB")
    else:
        budget = vram * a.mem_fraction - CUDA_CONTEXT
        fixed = dense + lm_head + embed + draft + kv + lin + GRAPHS + WORKSPACE
    left = budget - fixed
    e = model.get("experts")
    hot = 0
    slot = e["bytes_per_expert_mean"] / tp if e else 0  # the experts are tensor-parallel too: a rank holds 1/tp of a slot
    if e and left > 0:
        hot = int(left // slot // e["moe_layers"])
    # KT_HOT_KV_LEND: the empty pool's pages carry slots as well (~90% of the pool), given back as the KV grows;
    # the fixed KT_GPU_EXPERTS is the plan minus those (the TP1 default takes them, TP2 measured no gain)
    lent = int(kv * 0.9 // slot // e["moe_layers"]) if e and slot else 0
    # CPU pools: one node per rank at TP2 (rank i on node i's card), the card's node at TP1
    pools = []
    for g in ranks:
        n = [x for x in host["numa"] if x["node"] == g["numa_node"]]
        pools.append({"node": g["numa_node"], "threads": max(4, n[0]["physical_cores"] - 2) if n else 0,
                      "ram_free": n[0]["mem_free"] if n else 0, "page_cache": n[0]["file_pages"] if n else 0})
    threads = sum(p["threads"] for p in pools)
    ram_need = e["bytes_total"] + model["ple_bytes"] if e else model["ple_bytes"]
    # the prefill chunk from what stays free after the pool and the hot slots (the lend returns hot VRAM for the
    # prefill, so the scratch may also take the hot slots' share: counted at half)
    hot_bytes = hot * slot * (e["moe_layers"] if e else 0)
    free_after = left - hot_bytes
    chunk = int(min(CHUNK_MAX, max(2048, (free_after + hot_bytes / 2) // PREFILL_SCRATCH_PER_TOKEN)))
    chunk -= chunk % 1024
    for p in pools:
        if p["page_cache"] > 32 * GiB:
            notes.append(f"node {p['node']} holds {p['page_cache'] / GiB:.0f} GiB of page cache: drop it before the launch (another model's files) or the arenas spill to the other node")
    if ram_need > sum(n["mem_total"] for n in host["numa"]) * 0.9:
        notes.append(f"experts + PLE need {ram_need / GiB:.0f} GiB of RAM against {sum(n['mem_total'] for n in host['numa']) / GiB:.0f}")
    m = host["memory"]
    if m["swap_total"]:
        notes.append("swap is on: swapoff before a streamed-prefill launch")
    if m["thp_enabled"] != "madvise":
        notes.append(f"THP enabled={m['thp_enabled']}: madvise expected")
    if m["min_free_kbytes"] < 4 * 1024 * 1024:
        notes.append(f"min_free_kbytes {m['min_free_kbytes'] // 1024} MiB: 8 GiB on this host for the pinned arenas")
    busy = host.get("gpu_in_use", [])
    if busy:
        notes.append("GPUs in use: " + ", ".join(f"{u['name']}({u['pid']})" for u in busy))
    return {"tp": tp, "gpus": [g["uuid"] for g in ranks], "gpu_nodes": [g["numa_node"] for g in ranks],
            "vram_per_rank": vram, "mem_fraction": a.mem_fraction, "budget": budget,
            "terms": ({"weights_fixed": cal["weights_fixed"], "draft": 0 if cal.get("includes_draft") else draft, "kv_pool": kv, "linear_states": lin, "overhead": cal["overhead"]} if cal else
                      {"dense+hc": dense, "lm_head": lm_head, "token_table": embed, "draft": draft, "kv_pool": kv,
                       "linear_states": lin, "graphs": GRAPHS, "workspace": WORKSPACE, "cuda_context": CUDA_CONTEXT}),
            "pool_predicted_tokens": int((budget - fixed + kv - hot * slot * (e["moe_layers"] if e else 0)) / max(1, kv / max(1, a.context))) if cal else None,
            "left_for_hot": left, "hot_experts_per_layer": hot, "lent_from_empty_pool": lent, "moe_layers": e["moe_layers"] if e else 0,
            "hot_fixed_with_kv_lend": max(0, hot - lent), "expert_slot_per_rank": slot,
            "context": a.context, "kv_dtype": a.kv_dtype, "mamba_slots": a.mamba_slots, "chunk": chunk,
            "cpu_pools": pools, "threads": threads, "ram_need_experts_ple": ram_need, "notes": notes}


def table(p):
    g = lambda x: f"{x / GiB:6.2f} GiB"
    lines = [f"tp {p['tp']} on nodes {p['gpu_nodes']}: VRAM {g(p['vram_per_rank'])} x {p['mem_fraction']} - context = budget {g(p['budget'])} per rank"]
    for k, v in p["terms"].items():
        lines.append(f"  {k:14s} {g(v)}")
    lines.append(f"  left for hot  {g(p['left_for_hot'])} -> {p['hot_experts_per_layer']} hot experts per layer (slot {p['expert_slot_per_rank'] / 2**20:.2f} MiB per rank); "
                 f"with KT_HOT_KV_LEND the empty pool lends ~{p['lent_from_empty_pool']} more, so KT_GPU_EXPERTS={p['hot_fixed_with_kv_lend']} then")
    lines.append(f"context {p['context']} ({p['kv_dtype']} KV, pool = context), mamba slots {p['mamba_slots']}, prefill chunk {p['chunk']}"
                 + (f"; pool predicted at this hot: {p['pool_predicted_tokens']} tokens" if p.get("pool_predicted_tokens") else ""))
    lines.append("cpu pools: " + ", ".join(f"node {q['node']} x {q['threads']} threads (free {q['ram_free'] / GiB:.0f} GiB, page cache {q['page_cache'] / GiB:.0f})" for q in p["cpu_pools"])
                 + f"; experts + PLE in RAM {g(p['ram_need_experts_ple'])}")
    for n in p["notes"]:
        lines.append(f"! {n}")
    return "\n".join(lines)


def env_line(p):
    # the names your launcher maps to sglang's --tp, --kt-num-gpu-experts, --mem-fraction-static, --context-length,
    # --chunked-prefill-size, --kt-cpuinfer, --max-mamba-cache-size, --kv-cache-dtype
    return (f"TP={p['tp']} KT_GPU_EXPERTS={p['hot_experts_per_layer']} KT_GPU_EXPERTS_WITH_KV_LEND={p['hot_fixed_with_kv_lend']} "
            f"MEM_FRACTION={p['mem_fraction']} CONTEXT={p['context']} CHUNK={p['chunk']} KT_CPU_THREADS={p['threads']} "
            f"MAMBA_SLOTS={p['mamba_slots']} KV_DTYPE={'fp8_e4m3' if p['kv_dtype'] == 'fp8' else 'bfloat16'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--gguf")
    ap.add_argument("--experts-from")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--mamba-slots", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.93)
    ap.add_argument("--embed-host", type=int, default=1)
    ap.add_argument("--draft-bytes", type=float)
    ap.add_argument("--host-json")
    ap.add_argument("--model-json")
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--constants", help="a profile JSON with \"constants\" from plan-calibrate.py, or a JSON of the constants")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    a.constants_dict = None
    if a.constants:
        cj = json.load(open(a.constants))
        a.constants_dict = cj.get("constants", cj)
    host = load([sys.executable, os.path.join(HERE, "hostprobe.py"), "--json"], a.host_json)
    mcmd = [sys.executable, os.path.join(HERE, "modelprobe.py"), a.model_dir, "--json"] + (["--gguf", a.gguf] if a.gguf else []) \
        + (["--experts-from", a.experts_from] if a.experts_from else [])
    model = load(mcmd, a.model_json)
    p = plan(host, model, a)
    if a.env:
        print(env_line(p))
    elif a.json:
        print(json.dumps(p, indent=1))
    else:
        print(table(p))


if __name__ == "__main__":
    sys.exit(main())
