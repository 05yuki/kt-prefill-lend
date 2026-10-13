#!/usr/bin/env python3
"""Serve a model with its patched tree: probe the host, size the launch, set the profile's knobs, start SGLang.

usage: serve.py --profile NAME --venv VENV --tree SGLANG_TREE --config-dir DIR [--experts DIR | --gguf DIR]
                [--draft-dir DIR] [--tp 1|2] [--context T] [--port 8080] [--hot N] [--mem-fraction F] [--chunk C]
                [--kv-dtype bf16|fp8] [--mamba-slots N] [--threads N] [--placement FILE] [--hot-seed JSON]
                [--draft-hot FILE] [--log-dir DIR] [--cache-dir DIR] [--env KEY=VALUE ...] [--background] [--dry-run]

  --profile      profiles/<NAME>.json: the command line, the env defaults (per TP) and how the weights are laid out
                 (flash-next-gguf: everything from a GGUF; glm53-flash: fp8 dense + NVFP4 experts)
  --config-dir   the model directory SGLang loads (config + tokenizer; for a GGUF serve, tools/make-config-dir.py)
  --gguf / --experts   where the experts are (a GGUF dir, or a second checkpoint dir); default: the config dir
  --draft-dir    the MTP draft directory when the profile serves one (tools/make-draft-dir.py)
  --hot / --mem-fraction / --chunk / --threads   override the plan (tools/plan-launch.py prints the arithmetic)
The server log and a manifest (command, env, plan, package versions) go to --log-dir/<timestamp>/.
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(os.path.dirname(HERE), "profiles")


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd[:3])} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def wait_for_pool(log_path, proc, timeout=1800):
    """The pool the server sized (max_total_num_tokens=N in its log), or None if it died / timed out."""
    import re
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(log_path):
            with open(log_path, "rb") as f:
                m = re.search(rb"max_total_num_tokens=(\d+)", f.read())
            if m:
                return int(m.group(1))
        if proc.poll() is not None:
            return None
        time.sleep(3)
    return None


def model_layers(config_dir):
    c = json.load(open(os.path.join(config_dir, "config.json")))
    t = c.get("text_config", c)
    return int(t.get("num_hidden_layers") or 1)


def fill(items, vals):
    out = []
    for it in items:
        s = it
        for k, v in vals.items():
            s = s.replace("{" + k + "}", str(v))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="flash-next-gguf")
    ap.add_argument("--venv", required=True)
    ap.add_argument("--tree", required=True, help="the patched SGLang tree (python/ on PYTHONPATH)")
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--experts")
    ap.add_argument("--draft-dir")
    ap.add_argument("--tp", type=int)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--hot", type=int)
    ap.add_argument("--mem-fraction", type=float)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8"])
    ap.add_argument("--mamba-slots", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--placement")
    ap.add_argument("--hot-seed")
    ap.add_argument("--draft-hot")
    ap.add_argument("--served-model-name")
    ap.add_argument("--log-dir", default=os.path.join(os.getcwd(), "logs"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/kt-serve"))
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, overrides any default")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="launch, read the pool the server sized (max_total_num_tokens); if it is short of the context, "
                         "turn the shortfall into hot experts, remember the correction for this profile and key, and relaunch")
    a = ap.parse_args()

    pf = a.profile if a.profile.endswith(".json") else os.path.join(PROFILES, a.profile + ".json")
    prof = json.load(open(pf))
    w = prof["weights"]
    for k in ("venv", "tree", "config_dir", "gguf", "experts", "draft_dir", "placement", "hot_seed", "draft_hot", "log_dir", "cache_dir"):
        v = getattr(a, k)
        if v:
            setattr(a, k, os.path.abspath(os.path.expanduser(v)))
    py = os.path.join(a.venv, "bin", "python")
    for p, what in ((py, "venv python"), (os.path.join(a.tree, "python", "sglang"), "sglang tree"),
                    (os.path.join(a.config_dir, "config.json"), "config dir")):
        if not os.path.exists(p):
            raise SystemExit(f"{what} not found: {p}")
    experts_dir = a.gguf or a.experts or a.config_dir
    if w.get("gguf") and not a.gguf:
        raise SystemExit(f"profile {prof['name']} serves from a GGUF: pass --gguf DIR")
    if w.get("draft") == "dir" and a.draft_dir and not os.path.exists(os.path.join(a.draft_dir, "kt_mtp_gguf.json")):
        raise SystemExit(f"draft dir without kt_mtp_gguf.json: {a.draft_dir}")
    os.makedirs(a.cache_dir, exist_ok=True)
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = os.path.dirname(os.path.dirname(nvcc)) if nvcc else "/usr/local/cuda"

    # --- host, defaults for this TP, model, plan
    host = json.loads(sh([sys.executable, os.path.join(HERE, "hostprobe.py"), "--json", "--mounts", experts_dir]))
    if host["gpu_in_use"]:
        raise SystemExit("GPUs in use: " + ", ".join(f"{u['name']}({u['pid']})" for u in host["gpu_in_use"]))
    gpus = sorted(host["gpus"], key=lambda g: g["numa_node"])
    tp = a.tp or int(sorted(prof["defaults"])[0])
    if str(tp) not in prof["defaults"]:
        raise SystemExit(f"profile {prof['name']} has no measured defaults for tp {tp} (has {list(prof['defaults'])})")
    if len(gpus) < tp:
        raise SystemExit(f"{len(gpus)} GPUs, tp {tp}")
    d = prof["defaults"][str(tp)]
    context = a.context or d["context"]
    # the model's --context-length when the profile keeps it apart from the pool (DeepSeek V4: 1M over a 262K pool)
    context_length = max(context, d.get("context_length") or 0) if not a.context else context
    mem_fraction = a.mem_fraction or d["mem_fraction"]
    kv_dtype = a.kv_dtype or d["kv_dtype"]
    mamba_slots = a.mamba_slots or d["mamba_slots"]
    pools = d["pools"]
    ranks = gpus[:tp]
    plan_cmd = [sys.executable, os.path.join(HERE, "plan-launch.py"), a.config_dir, "--tp", str(tp), "--context", str(context),
                "--mem-fraction", str(mem_fraction), "--kv-dtype", kv_dtype, "--mamba-slots", str(mamba_slots),
                "--embed-host", str(w.get("embed_host", 0)), "--json"]
    if a.gguf:
        plan_cmd += ["--gguf", a.gguf]
    elif a.experts and w.get("experts_from"):
        plan_cmd += ["--experts-from", a.experts]
    if w.get("draft") == "none" or not a.draft_dir:
        plan_cmd += ["--draft-bytes", "0"]
    elif w.get("draft_bytes_assumed"):
        plan_cmd += ["--draft-bytes", str(int(w["draft_bytes_assumed"] * 2**30 / tp))]  # the draft dir's share per rank
    if prof.get("constants"):
        plan_cmd += ["--constants", pf]  # calibrated from a launch log (tools/plan-calibrate.py)
    plan = json.loads(sh(plan_cmd))
    kv_lend = prof.get("env_by_tp", {}).get(str(tp), {}).get("KT_HOT_KV_LEND") == "1"
    key = f"tp{tp}-ctx{context}-mamba{mamba_slots}-{os.path.basename(experts_dir.rstrip('/'))}"
    corr_path = os.path.join(a.cache_dir, "plan-corrections", f"{prof['name']}-{key}.json")
    corr = json.load(open(corr_path)) if os.path.exists(corr_path) else {}
    if a.hot is not None:
        hot = a.hot
    elif "hot" in d:
        hot = d["hot"]  # the profile's measured value; the plan is printed beside it
    else:
        hot = plan["hot_fixed_with_kv_lend"] if kv_lend else plan["hot_experts_per_layer"]
        hot = max(0, hot + corr.get("hot_delta", 0))  # learned from an earlier launch's pool (see --calibrate)
    chunk = a.chunk or d.get("chunk") or plan["chunk"]
    threads = a.threads or d.get("threads") or plan["threads"]
    vals = {"config_dir": a.config_dir, "experts_dir": experts_dir, "draft_dir": a.draft_dir or "", "port": a.port, "tp": tp,
            "numa_nodes": " ".join(str(g["numa_node"]) for g in ranks), "context": context, "context_length": context_length,
            "mem_fraction": mem_fraction,
            "kv_cache_dtype": "fp8_e4m3" if kv_dtype == "fp8" else "bfloat16", "mamba_slots": mamba_slots, "hot": hot,
            "threads": threads, "pools": pools, "chunk": chunk, "cache_dir": a.cache_dir, "key": key,
            "hot_dynamic": "1" if hot else "0", "served_model_name": a.served_model_name or prof["name"],
            "placement": a.placement or "", "hot_seed": a.hot_seed or "", "draft_hot": a.draft_hot or "", "draft_gguf": ""}
    if a.draft_dir and w.get("draft") == "dir":
        with open(os.path.join(a.draft_dir, "kt_mtp_gguf.json")) as f:
            vals["draft_gguf"] = json.load(f)["gguf"]

    # --- environment and command from the profile
    env = dict(os.environ)
    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(g["uuid"] for g in ranks),
                "CUDA_HOME": cuda_home,
                "PATH": os.path.join(a.venv, "bin") + os.pathsep + os.path.join(cuda_home, "bin") + os.pathsep + env.get("PATH", ""),
                "PYTHONPATH": os.path.join(a.tree, "python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(threads)})
    env.update(dict(zip(prof["env"].keys(), fill(prof["env"].values(), vals))))
    e_tp = prof.get("env_by_tp", {}).get(str(tp), {})
    env.update(dict(zip(e_tp.keys(), fill(e_tp.values(), vals))))
    for cond, block in (("draft_dir", "env_with_draft"), ("draft_hot", "env_with_draft_hot"), ("hot_seed", "env_with_hot_seed")):
        if getattr(a, cond) and block in prof:
            env.update(dict(zip(prof[block].keys(), fill(prof[block].values(), vals))))
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env = {k: v for k, v in env.items() if v != ""}
    cmd = [py]
    for arg in fill(prof["command"], vals):
        cmd += arg.split(" ") if arg == vals["numa_nodes"] else [arg]  # "--numa-node 0 1": one placeholder, two values
    if a.placement and "command_with_placement" in prof:
        cmd += fill(prof["command_with_placement"], vals)
    if a.draft_dir and "command_with_draft" in prof:
        cmd += fill(prof["command_with_draft"], vals)

    run = os.path.join(a.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    manifest = {"profile": prof["name"], "command": cmd,
                "env": {k: v for k, v in env.items() if k.startswith(("KT_", "SGLANG_", "CUDA_", "OMP_", "NCCL_", "TORCH"))},
                "plan": plan, "host": {k: host[k] for k in ("hostname", "kernel", "driver", "cuda_toolkit")}, "packages": {}}
    for n in ("torch", "flashinfer-python", "triton", "kt-kernel", "sglang"):
        try:
            manifest["packages"][n] = md.version(n)
        except md.PackageNotFoundError:
            manifest["packages"][n] = None
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{prof['name']} tp {tp}: hot {hot} per layer (plan {plan['hot_experts_per_layer']}, lent ~{plan['lent_from_empty_pool']}), "
          f"chunk {chunk}, threads {threads} on nodes {[g['numa_node'] for g in ranks]}, context {context}, mem {mem_fraction}, KV {kv_dtype}")
    for n in plan["notes"]:
        print("! " + n)
    print("log directory:", run)
    if a.dry_run:
        print(" ".join(cmd))
        return 0
    log = open(os.path.join(run, "server.log"), "ab")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    with open(os.path.join(run, "server.pid"), "w") as f:
        f.write(str(proc.pid))
    if a.calibrate:
        pool = wait_for_pool(os.path.join(run, "server.log"), proc)
        if pool is None:
            print("calibrate: the server did not report a pool (see the log)")
            return 1
        kv_tok = plan["terms"]["kv_pool"] / max(1, context)
        slot = plan["expert_slot_per_rank"] * (plan.get("moe_layers") or model_layers(a.config_dir))
        short = context - pool
        delta = -int(-(-short * kv_tok // slot)) if short > 0 else 0  # slots a layer the shortfall is worth, rounded up
        print(f"calibrate: pool {pool} against context {context}: {'short by ' + str(short) + ' tokens' if short > 0 else 'enough'}"
              f" -> hot delta {delta} (this launch used hot {hot})")
        os.makedirs(os.path.dirname(corr_path), exist_ok=True)
        with open(corr_path, "w") as f:
            json.dump({"hot_delta": corr.get("hot_delta", 0) + delta, "pool": pool, "hot": hot, "context": context}, f)
        if delta:
            proc.terminate()
            proc.wait(timeout=120)
            print(f"calibrate: relaunching with hot {hot + delta} (correction saved to {corr_path})")
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + [x for x in sys.argv[1:] if x != "--calibrate"])
        print("calibrate: the plan holds; correction saved")
    if a.background:
        print(f"started pid {proc.pid}")
        print(f"tail the log for 'max_total_num_tokens=': the pool the server sized against context {context}")
        return 0
    try:
        while proc.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        proc.terminate()
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
