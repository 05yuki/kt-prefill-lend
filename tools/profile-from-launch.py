#!/usr/bin/env python3
"""A serve profile from a launch that is running (or ran): the server process's environment and command line as
they are, with the paths and the sized values turned into placeholders. No transcription from a launcher script.

usage: profile-from-launch.py --pid PID | --environ FILE --command FILE   --name NAME --out profiles/NAME.json
         [--config-dir DIR] [--experts DIR] [--draft-dir DIR] [--cache-dir DIR] [--weights JSON]
  --pid        a running server: /proc/PID/environ and /proc/PID/cmdline (the launch_server process)
  --environ    a NUL-separated env dump (`tr '\\0' '\\n' < /proc/PID/environ` works too), with --command (one arg per line)
  --config-dir / --experts / --draft-dir / --cache-dir   the values that become {config_dir} etc.
  --weights    the profile's weights block, e.g. '{"gguf": true, "experts_from": false, "embed_host": 1, "draft": "dir"}'
Kept as the profile's env: KT_*, SGLANG_*, NCCL_*, TORCH_EXTENSIONS_DIR, TORCHINDUCTOR_CACHE_DIR, OMP_NUM_THREADS
(the thread count becomes {threads}). CUDA_VISIBLE_DEVICES and PATH / PYTHONPATH are serve.py's. The sized values
(--tp, --numa-node, --context-length, --max-total-tokens, --mem-fraction-static, --chunked-prefill-size,
--max-prefill-tokens, --kt-num-gpu-experts, --kt-cpuinfer, --kt-threadpool-count, --max-mamba-cache-size,
--kv-cache-dtype, --port) become placeholders and the profile's defaults for that TP.
"""
import argparse
import json
import os
import sys

ENV_PREFIXES = ("KT_", "SGLANG_", "NCCL_", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR")
SIZED = {"--tp": "tp", "--tp-size": "tp", "--tensor-parallel-size": "tp", "--context-length": "context_length",
         "--max-total-tokens": "context", "--mem-fraction-static": "mem_fraction",
         "--chunked-prefill-size": "chunk", "--max-prefill-tokens": "chunk", "--kt-num-gpu-experts": "hot",
         "--kt-cpuinfer": "threads", "--kt-threadpool-count": "pools", "--max-mamba-cache-size": "mamba_slots",
         "--kv-cache-dtype": "kv_cache_dtype", "--port": "port", "--served-model-name": "served_model_name"}
DRAFT_ARGS = ("--speculative-draft-model-path", "--speculative-algorithm", "--speculative-num-steps", "--speculative-eagle-topk",
              "--speculative-num-draft-tokens")
PLACEMENT_ARGS = ("--init-expert-location", "--kt-expert-placement-strategy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--environ")
    ap.add_argument("--command")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir")
    ap.add_argument("--experts")
    ap.add_argument("--draft-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--weights", default='{"gguf": false, "experts_from": false, "embed_host": 0, "draft": "none"}')
    a = ap.parse_args()
    if a.pid:
        env_raw = open(f"/proc/{a.pid}/environ", "rb").read().decode("utf-8", "replace")
        cmd = open(f"/proc/{a.pid}/cmdline", "rb").read().decode("utf-8", "replace").split("\0")
    else:
        env_raw = open(a.environ, "rb").read().decode("utf-8", "replace")
        cmd = [l.rstrip("\n") for l in open(a.command) if l.strip()]
    sep = "\0" if "\0" in env_raw else "\n"
    env = dict(kv.split("=", 1) for kv in env_raw.split(sep) if "=" in kv)
    cmd = [c for c in cmd if c]
    # the command: python -m sglang.launch_server ... -> from "-m" on
    if "-m" in cmd:
        cmd = cmd[cmd.index("-m"):]
    paths = {k: os.path.abspath(v) for k, v in (("config_dir", a.config_dir), ("experts_dir", a.experts), ("draft_dir", a.draft_dir), ("cache_dir", a.cache_dir)) if v}

    def sub_paths(v):
        for k, p in sorted(paths.items(), key=lambda kv: -len(kv[1])):
            if v == p or v.startswith(p + "/"):
                return "{" + k + "}" + v[len(p):]
        return v

    defaults, command, draft_cmd, placement_cmd = {}, [], [], []
    i = 0
    while i < len(cmd):
        arg = cmd[i]
        if arg.startswith("--"):
            vals = []
            j = i + 1
            while j < len(cmd) and not cmd[j].startswith("--"):
                vals.append(cmd[j]); j += 1
            if arg in SIZED and vals:
                key = SIZED[arg]
                defaults.setdefault(key, vals[0])
                command += [arg, "{" + key + "}"]
            elif arg == "--numa-node":
                command += [arg, "{numa_nodes}"]
            elif arg in DRAFT_ARGS:
                draft_cmd += [arg] + [sub_paths(v) for v in vals]
            elif arg in PLACEMENT_ARGS:
                placement_cmd += [arg] + (["{placement}"] if arg == "--init-expert-location" else vals)
            else:
                command += [arg] + [sub_paths(v) for v in vals]
            i = j
        else:
            command.append(arg); i += 1
    threads = env.get("OMP_NUM_THREADS")
    penv = {}
    for k, v in env.items():
        if not k.startswith(ENV_PREFIXES) or k.startswith("KT_LAUNCH_TIMING"):
            continue  # the timing instrument is not part of the serve
        if k in ("KT_GPU_EXPERTS",):
            v = "{hot}"
        elif k == "KT_PREFILL_LEND_HINT":
            v = "{cache_dir}/lend-hint-{key}"
        elif k in ("KT_MTP_GGUF",):
            v = "{draft_gguf}"
        elif k in ("KT_DRAFT_HOT",):
            v = "{draft_hot}"
        elif k in ("KT_HOT_SEED",):
            v = "{hot_seed}"
        elif k in ("KT_EXPERT_PLACEMENT",) and v not in ("none", ""):
            v = "{placement}"
        else:
            v = sub_paths(v)
        penv[k] = v
    tp = defaults.get("tp", "1")
    # the KV pool is the context: --max-total-tokens holds it there
    if "--max-total-tokens" not in command:
        command += ["--max-total-tokens", "{context}"]
    # the context, the hot count and the memory fraction are not kept: serve.py plans them from the model and the
    # card (the context is the model's, the pool is the context, the hot experts take what is left)
    d = {"kv_dtype": "fp8" if defaults.get("kv_cache_dtype", "").startswith("fp8") else "bf16",
         "mamba_slots": int(defaults.get("mamba_slots", 8) or 8), "pools": int(defaults.get("pools", 1) or 1),
         "chunk": int(defaults.get("chunk", 0) or 0), "threads": int(threads) if threads else None,
         "free_min_gb": 1.0}
    prof = {"name": a.name, "description": f"captured from a launch ({'pid ' + str(a.pid) if a.pid else a.command})",
            "weights": json.loads(a.weights), "defaults": {tp: d}, "command": command,
            "env": dict(sorted(penv.items())), "env_by_tp": {tp: {}}, "min_hot": 0}
    if draft_cmd:
        prof["command_with_draft"] = draft_cmd
        for k in ("KT_NEXT_MTP_PAD_FP8", "KT_NEXT_MTP_GPU_ONLY", "KT_MTP_Q2_0", "KT_MTP_GGUF"):
            if k in prof["env"]:
                prof.setdefault("env_with_draft", {})[k] = prof["env"].pop(k)
    if placement_cmd:
        prof["command_with_placement"] = placement_cmd
    if "KT_DRAFT_HOT" in prof["env"]:
        prof["env_with_draft_hot"] = {"KT_DRAFT_HOT": prof["env"].pop("KT_DRAFT_HOT")}
    if "KT_HOT_SEED" in prof["env"]:
        prof["env_with_hot_seed"] = {"KT_HOT_SEED": prof["env"].pop("KT_HOT_SEED")}
    with open(a.out, "w") as f:
        json.dump(prof, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{a.out}: tp {tp}, {len(command)} command args, {len(prof['env'])} env knobs, defaults {d}")


if __name__ == "__main__":
    sys.exit(main())
