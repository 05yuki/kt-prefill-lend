#!/usr/bin/env python3
"""Constants for plan-launch from one launch log of the same profile, so the next plan predicts the pool and the hot
count without a second launch.

usage: plan-calibrate.py <server.log> --hot N --slot-bytes B --moe-layers L --vram-total BYTES --mem-fraction F
                         [--kv-dtype bf16|fp8] [--write profiles/NAME.json]
  --hot / --slot-bytes / --moe-layers   the hot experts the launch had and one slot's bytes on one rank
                                        (plan-launch prints `expert_slot_per_rank`), so the dense part of the
                                        weights can be separated from the hot experts
Reads SGLang's accounting lines: "Load weight end ... mem usage=X GB" (weights incl. hot experts), "KV Cache is
allocated. #tokens: N, KV size: S GB" (the pool and the real bytes a token takes, DSA index pools included),
"mamba_cache_size: N ... ssm_state size: S GB" / "conv_state size", "Memory pool end. avail mem=A GB", and
"Capture ... end ... mem usage=G GB". Derives: weights_fixed (everything on the GPU but the hot slots),
kv_bytes_per_token (at the log's KV dtype), states_bytes (all slots), overhead (what the server keeps beside the
pool: total * mem_fraction - weights - states - pool), graphs. Writes them as "constants" into the profile
(--write) for plan-launch's `--constants` / serve.py.
"""
import argparse
import json
import re
import sys

GiB = 2**30


def graphs_per_rank(s):
    """The CUDA graphs' memory of the rank that took most (a rank's captures summed; lines tagged TPn)."""
    per = {}
    for line in s.splitlines():
        m = re.search(r"Capture [^\n]*end\.[^\n]*mem usage=([0-9.]+) GB", line)
        if m:
            r = re.search(r"\bTP(\d+)\]", line)
            k = r.group(1) if r else "0"
            per[k] = per.get(k, 0.0) + float(m.group(1))
    return max(per.values()) if per else None


def parse(log):
    s = open(log, "rb").read().decode("utf-8", "replace")
    g = lambda pat, cast=float: (cast(re.search(pat, s).group(1)) if re.search(pat, s) else None)
    # every rank prints its own lines, and a draft model its own after the target's: the target's weights and the
    # draft's (MTP / NextN) on the rank that holds most, and the least any rank had free once every pool was made
    loads = re.findall(r"Load weight end\.[^\n]*?type=(\w+)[^\n]*mem usage=([0-9.]+) GB", s)
    target = [float(m) for t, m in loads if not t.endswith(("MTP", "NextN"))]
    draft = [float(m) for t, m in loads if t.endswith(("MTP", "NextN"))]
    pool_end = [float(x) for x in re.findall(r"Memory pool end\. avail mem=([0-9.]+) GB", s)]
    out = {
        "weights_gb": (max(target) + (max(draft) if draft else 0)) if target else g(r"Load weight end\.[^\n]*mem usage=([0-9.]+) GB"),
        "pool_tokens": g(r"KV Cache is allocated\. #tokens: (\d+)", int),
        "kv_gb": g(r"KV Cache is allocated\. #tokens: \d+, KV size: ([0-9.]+) GB"),
        "ssm_gb": g(r"ssm_state size: ([0-9.]+)GB"),
        "conv_gb": g(r"conv_state size: ([0-9.]+)GB"),
        "mamba_slots": g(r"mamba_cache_size: (\d+)", int),
        "avail_after_pool_gb": min(pool_end) if pool_end else None,
        "graphs_gb": graphs_per_rank(s),
        "max_total_num_tokens": g(r"max_total_num_tokens=(\d+)", int),
        "had_draft": bool(draft) or "speculative_draft_model_path': '/" in s or "speculative_draft_model_path\": \"/" in s,
    }
    # Flash-Next logs: K size / V size lines (one per pool: the target's full-attention pool first, the draft's
    # after; every rank prints its own, so take the lines of one rank: up to the first repeat of the pool size)
    if out["kv_gb"] is None:
        # every pool a rank makes (the target's, a draft's) prints one line, tagged TPn: summed per rank, the
        # largest rank kept (lines of the ranks interleave, so the order says nothing)
        per, pool = {}, None
        for line in s.splitlines():
            m = re.search(r"KV Cache is allocated\.[^\n]*#tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB", line)
            if m:
                r = re.search(r"\bTP(\d+)\]", line)
                k = r.group(1) if r else "0"
                per[k] = per.get(k, 0.0) + float(m.group(2)) + float(m.group(3))
                pool = int(m.group(1)) if pool is None else pool
        if per:
            out["pool_tokens"] = pool
            out["kv_gb"] = max(per.values())
    # DeepSeek V4 logs: the hybrid pools are sized from one "bytes per full token" figure
    # ("DSV4 memory calculation: bytes_per_full_token=B ... full_token=N" then "DSV4 pool sizes: full=N, swa=..."):
    # the last pool-sizes line is the pool the serve kept (the first is the uncapped capacity)
    if out["kv_gb"] is None:
        bpt = re.findall(r"bytes_per_full_token=([0-9.]+)", s)
        pools = re.findall(r"DSV4 pool sizes: full=(\d+), swa=(\d+), c4=(\d+), c128=(\d+)", s)
        if bpt and pools:
            full, swa, c4, c128 = (int(x) for x in pools[-1])
            out["pool_tokens"] = full
            out["kv_gb"] = float(bpt[-1]) * full / 1e9
            out["dsv4_pools"] = {"full": full, "swa": swa, "c4": c4, "c128": c128, "bytes_per_full_token": float(bpt[-1])}
    out["pool_capped"] = None  # set by main() when the pool equals the context
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--hot", type=int, required=True)
    ap.add_argument("--slot-bytes", type=float, required=True, help="one expert slot on one rank, bytes")
    ap.add_argument("--moe-layers", type=int, required=True)
    ap.add_argument("--vram-total", type=float, required=True, help="bytes of one card")
    ap.add_argument("--mem-fraction", type=float, required=True)
    ap.add_argument("--kv-dtype", default="bf16")
    ap.add_argument("--tp", type=int, required=True, help="the launch's TP: the constants are per rank, for that TP only")
    ap.add_argument("--context", type=int, help="the launch's context; when the pool equals it, the overhead is only an upper bound")
    ap.add_argument("--write")
    a = ap.parse_args()
    p = parse(a.log)
    missing = [k for k in ("weights_gb", "pool_tokens", "kv_gb") if p[k] is None]
    if missing:
        raise SystemExit(f"{a.log}: no {missing} lines")
    hot_bytes = a.hot * a.slot_bytes * a.moe_layers
    weights = p["weights_gb"] * GiB
    states = ((p["ssm_gb"] or 0) + (p["conv_gb"] or 0)) * GiB
    pool_bytes = p["kv_gb"] * GiB
    capped = a.context is not None and p["pool_tokens"] >= a.context
    if not capped or p["avail_after_pool_gb"] is None:
        # SGLang sized the pool itself: static less everything else is what it kept beside them, its own
        # reservations included (exact when the pool was not capped by the context)
        overhead = a.vram_total * a.mem_fraction - weights - states - pool_bytes
    else:
        # a capped pool says only what the card had free once it was made: a lower bound on what SGLang keeps
        # (0.5 GiB short on Flash-Next GGUF TP2); the next launch's shortfall, if any, corrects it
        overhead = a.vram_total - p["avail_after_pool_gb"] * GiB - weights - states - pool_bytes
    c = {"weights_fixed": weights - hot_bytes, "kv_bytes_per_token": pool_bytes / p["pool_tokens"], "kv_dtype": a.kv_dtype,
         "states_bytes": states, "states_slots": p["mamba_slots"], "overhead": overhead, "graphs": (p["graphs_gb"] or 0) * GiB,
         "overhead_is_upper_bound": capped, "includes_draft": bool(p["had_draft"]),
         **({"dsv4_pools": p["dsv4_pools"]} if p.get("dsv4_pools") else {}),
         "from": {"log": a.log, "tp": a.tp, "hot": a.hot, "mem_fraction": a.mem_fraction, "pool_tokens": p["pool_tokens"],
                  "avail_after_pool_gb": p["avail_after_pool_gb"]}}
    print(f"weights {p['weights_gb']:.2f} GB of which hot {hot_bytes / GiB:.2f} GiB -> fixed {c['weights_fixed'] / GiB:.2f} GiB; "
          f"pool {p['pool_tokens']} tokens = {p['kv_gb']:.2f} GB -> {c['kv_bytes_per_token'] / 1024:.2f} KiB a token ({a.kv_dtype}); "
          f"states {states / GiB:.2f} GiB ({p['mamba_slots']} slots); overhead {overhead / GiB:.2f} GiB{' (upper bound: the pool was capped by the context)' if capped else ''}; graphs {c['graphs'] / GiB:.2f} GiB")
    if a.write:
        prof = json.load(open(a.write))
        old = prof.get("constants") or {}
        if capped and old.get("overhead", 0) > c["overhead"] and old.get("from", {}).get("tp") == a.tp:
            # a lower bound never replaces what an uncapped launch measured
            c["overhead"] = old["overhead"]
            print(f"overhead kept at {old['overhead'] / GiB:.2f} GiB from {old['from'].get('log')}")
        prof["constants"] = c
        with open(a.write, "w") as f:
            json.dump(prof, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"written to {a.write}")
    else:
        print(json.dumps(c, indent=1))


if __name__ == "__main__":
    sys.exit(main())
