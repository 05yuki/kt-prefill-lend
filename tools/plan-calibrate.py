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


def parse(log):
    s = open(log, "rb").read().decode("utf-8", "replace")
    g = lambda pat, cast=float: (cast(re.search(pat, s).group(1)) if re.search(pat, s) else None)
    out = {
        "weights_gb": g(r"Load weight end\.[^\n]*mem usage=([0-9.]+) GB"),
        "pool_tokens": g(r"KV Cache is allocated\. #tokens: (\d+)", int),
        "kv_gb": g(r"KV Cache is allocated\. #tokens: \d+, KV size: ([0-9.]+) GB"),
        "ssm_gb": g(r"ssm_state size: ([0-9.]+)GB"),
        "conv_gb": g(r"conv_state size: ([0-9.]+)GB"),
        "mamba_slots": g(r"mamba_cache_size: (\d+)", int),
        "avail_after_pool_gb": g(r"Memory pool end\. avail mem=([0-9.]+) GB"),
        "graphs_gb": sum(float(x) for x in re.findall(r"Capture [^\n]*end\.[^\n]*mem usage=([0-9.]+) GB", s)) or None,
        "max_total_num_tokens": g(r"max_total_num_tokens=(\d+)", int),
        "had_draft": "speculative_draft_model_path': '/" in s or "speculative_draft_model_path\": \"/" in s,
    }
    # Flash-Next logs: K size / V size lines (one per pool: the target's full-attention pool first, the draft's
    # after; every rank prints its own, so take the lines of one rank: up to the first repeat of the pool size)
    if out["kv_gb"] is None:
        ks = re.findall(r"KV Cache is allocated\.[^\n]*#tokens: (\d+), K size: ([0-9.]+) GB, V size: ([0-9.]+) GB", s)
        if ks:
            out["pool_tokens"] = int(ks[0][0])
            seen, total = set(), 0.0
            for n, k, v in ks:
                if (n, k, v) in seen:
                    break
                seen.add((n, k, v))
                total += float(k) + float(v)
            out["kv_gb"] = total
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
    budget = a.vram_total * a.mem_fraction
    overhead = budget - weights - states - pool_bytes
    capped = a.context is not None and p["pool_tokens"] >= a.context
    c = {"weights_fixed": weights - hot_bytes, "kv_bytes_per_token": pool_bytes / p["pool_tokens"], "kv_dtype": a.kv_dtype,
         "states_bytes": states, "states_slots": p["mamba_slots"], "overhead": overhead, "graphs": (p["graphs_gb"] or 0) * GiB,
         "overhead_is_upper_bound": capped, "includes_draft": bool(p["had_draft"]),
         "from": {"log": a.log, "hot": a.hot, "mem_fraction": a.mem_fraction, "pool_tokens": p["pool_tokens"],
                  "avail_after_pool_gb": p["avail_after_pool_gb"]}}
    print(f"weights {p['weights_gb']:.2f} GB of which hot {hot_bytes / GiB:.2f} GiB -> fixed {c['weights_fixed'] / GiB:.2f} GiB; "
          f"pool {p['pool_tokens']} tokens = {p['kv_gb']:.2f} GB -> {c['kv_bytes_per_token'] / 1024:.2f} KiB a token ({a.kv_dtype}); "
          f"states {states / GiB:.2f} GiB ({p['mamba_slots']} slots); overhead {overhead / GiB:.2f} GiB{' (upper bound: the pool was capped by the context)' if capped else ''}; graphs {c['graphs'] / GiB:.2f} GiB")
    if a.write:
        prof = json.load(open(a.write))
        prof["constants"] = c
        with open(a.write, "w") as f:
            json.dump(prof, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"written to {a.write}")
    else:
        print(json.dumps(c, indent=1))


if __name__ == "__main__":
    sys.exit(main())
