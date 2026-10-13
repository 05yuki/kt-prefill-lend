#!/usr/bin/env python3
"""Serve a Flash-Next GGUF with the patched tree: probe the host, size the launch, set the knobs, start SGLang.

usage: serve.py --venv VENV --tree SGLANG_TREE --config-dir DIR --gguf DIR [--draft-dir DIR]
                [--tp 1|2] [--context T] [--port 8080] [--hot N] [--mem-fraction F] [--chunk C]
                [--kv-dtype bf16|fp8] [--mamba-slots 8] [--threads N] [--placement JSON] [--log-dir DIR]
                [--env KEY=VALUE ...] [--background] [--dry-run]

  --config-dir   the model's config + tokenizer directory (tools/make-config-dir.py)
  --gguf         the GGUF directory (experts, dense, PLE: the one file set serves everything)
  --draft-dir    the MTP draft directory (tools/make-draft-dir.py); without it, no speculative decoding
  --hot / --mem-fraction / --chunk / --threads  override the plan (tools/plan-launch.py prints the arithmetic)
Everything else is the measured default of the serving tree (native-ubuntu/GGUF-EXPERTS in the serving repo):
TP1 = one card + its NUMA node's cores, bf16 hc mix, KV-page lend on; TP2 = both cards, both nodes, the int8 hc
mix, the prefill-only lend. The server log and a manifest (command, env, plan, package versions) go to --log-dir.
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


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--tree", required=True, help="the patched SGLang tree (python/ on PYTHONPATH)")
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--draft-dir")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--hot", type=int)
    ap.add_argument("--mem-fraction", type=float)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--mamba-slots", type=int, default=8)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--placement", help="expert placement JSON (--init-expert-location); default: none")
    ap.add_argument("--hot-seed", help="hot-cache seed JSON from an earlier run (KT_HOT_SEED)")
    ap.add_argument("--draft-hot", help="token list for the draft's reduced head (tools/make-draft-hot-tokens.py, KT_DRAFT_HOT)")
    ap.add_argument("--served-model-name", default="flash-next-gguf")
    ap.add_argument("--log-dir", default=os.path.join(os.getcwd(), "logs"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/kt-serve"),
                    help="dequantized dense weights and the CuTe DSL warmup cache (launch time)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, overrides any default")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for k in ("venv", "tree", "config_dir", "gguf", "draft_dir", "placement", "hot_seed", "draft_hot", "log_dir", "cache_dir"):
        v = getattr(a, k)
        if v:
            setattr(a, k, os.path.abspath(os.path.expanduser(v)))
    py = os.path.join(a.venv, "bin", "python")
    os.makedirs(a.cache_dir, exist_ok=True)
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = os.path.dirname(os.path.dirname(nvcc)) if nvcc else "/usr/local/cuda"
    for p, what in ((py, "venv python"), (os.path.join(a.tree, "python", "sglang"), "sglang tree"),
                    (os.path.join(a.config_dir, "config.json"), "config dir"), (a.gguf, "gguf dir")):
        if not os.path.exists(p):
            raise SystemExit(f"{what} not found: {p}")

    # --- host and model, and the plan
    host = json.loads(sh([sys.executable, os.path.join(HERE, "hostprobe.py"), "--json", "--mounts", a.gguf]).stdout)
    if host["gpu_in_use"]:
        raise SystemExit("GPUs in use: " + ", ".join(f"{u['name']}({u['pid']})" for u in host["gpu_in_use"]))
    gpus = sorted(host["gpus"], key=lambda g: g["numa_node"])
    if len(gpus) < a.tp:
        raise SystemExit(f"{len(gpus)} GPUs, tp {a.tp}")
    tp1 = a.tp == 1
    context = a.context or (65536 if tp1 else 131072)
    mem_fraction = a.mem_fraction or (0.93 if tp1 else 0.955)
    plan_cmd = [sys.executable, os.path.join(HERE, "plan-launch.py"), a.config_dir, "--gguf", a.gguf, "--tp", str(a.tp),
                "--context", str(context), "--mem-fraction", str(mem_fraction), "--kv-dtype", a.kv_dtype,
                "--mamba-slots", str(a.mamba_slots), "--json"]
    plan = json.loads(sh(plan_cmd).stdout)
    hot = a.hot if a.hot is not None else (plan["hot_fixed_with_kv_lend"] if tp1 else plan["hot_experts_per_layer"])
    chunk = a.chunk or plan["chunk"]
    threads = a.threads or plan["threads"]
    pools = 1 if tp1 else 2
    ranks = gpus[: a.tp]

    # --- environment: the serving tree's measured defaults for the GGUF stack
    env = dict(os.environ)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(g["uuid"] for g in ranks),
        "CUDA_HOME": cuda_home,
        "PATH": os.path.join(a.venv, "bin") + os.pathsep + os.path.join(cuda_home, "bin") + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": os.path.join(a.tree, "python") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "PYTHONUNBUFFERED": "1",
        "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1", "SGLANG_NUMA_BIND_V2": "True",
        "OMP_NUM_THREADS": str(threads),
        # the whole model from the GGUF
        "KT_GGUF_DENSE": a.gguf, "KT_GGUF_DENSE_MODEL_PATH": a.config_dir, "KT_GGUF_DENSE_QUANT": "1",
        "KT_GGUF_FAST_HEADER": "1", "KT_FAST_PINNED": "1", "KT_GGUF_NOCOPY": "1", "KT_GGUF_DENSE_THREADS": "8",
        "KT_GGUF_PREFETCH": "8", "KT_EMBED_HOST": "1",
        "KT_GGUF_DENSE_CACHE": os.path.join(a.cache_dir, "dense"), "KT_CUTEDSL_CACHE": os.path.join(a.cache_dir, "cutedsl"),
        # the streamer (GGUF edition) and the lend
        "KT_GPU_STREAM_PREFILL": "256", "KT_GPU_STREAM_GROUP": "64", "KT_GPU_STREAM_EARLY_INIT": "1",
        "KT_STREAM_MEMOPS": "1", "KT_PREFILL_LEND": "1", "KT_PREFILL_LEND_CALIBRATE": "0",
        # the hot cache
        "KT_GPU_EXPERTS": str(hot), "KT_EXPERT_PLACEMENT": "none", "KT_HOT_DYNAMIC": "1" if hot else "0",
        "KT_HOT_INTERVAL": "64", "KT_HOT_SWAPS": "4", "KT_LLAMAFILE_GROUP_MIN_LEN": "2",
        "KT_HOT_KV_LEND": "1" if tp1 else "0",
        # the hyper-connection mix: bf16 at TP1 (identical weights), int8 + prefill-only lend at TP2
        "KT_HC_FP8": "0" if tp1 else "int8", "KT_HC_FP8_ALL": "0" if tp1 else "1",
        "KT_PREFILL_LEND_PFONLY": "" if tp1 else "input_mix_weight",
        # TP2 decode all-reduce through host memory
        "SGLANG_LL_ALLREDUCE": "0" if tp1 else "1",
        "KT_TASKQUEUE_SPIN_US": "200",
        "KT_EXPERT_SHM": "0",  # LLAMAFILE experts: no memfd arenas
        "SGLANG_HOST_STAGED_ALLREDUCE": "1",  # prefill all-reduce staged through host memory (TP2; idle at TP1)
        "KT_GPU_STREAM_DROP_CACHE": a.config_dir,
        # the lend's chunk calibration cache, keyed like the serving launcher
        "KT_PREFILL_LEND_HINT": os.path.join(a.cache_dir, f"lend-hint-tp{a.tp}-ctx{context}-mamba{a.mamba_slots}-{os.path.basename(a.gguf.rstrip('/'))}"),
    })
    if a.draft_hot:
        env["KT_DRAFT_HOT"] = a.draft_hot
    if a.hot_seed:
        env["KT_HOT_SEED"] = a.hot_seed
    if a.draft_dir:
        with open(os.path.join(a.draft_dir, "kt_mtp_gguf.json")) as f:
            env["KT_MTP_GGUF"] = json.load(f)["gguf"]
        env.update({"KT_NEXT_MTP_GPU_ONLY": "1", "KT_MTP_Q2_0": "1"})
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env = {k: v for k, v in env.items() if v != ""}

    # --- the command
    cmd = [py, "-m", "sglang.launch_server",
           "--model-path", a.config_dir, "--served-model-name", a.served_model_name,
           "--model-impl", "sglang", "--load-format", "safetensors", "--dtype", "bfloat16",
           "--host", "0.0.0.0", "--port", str(a.port), "--tp", str(a.tp),
           "--numa-node", *[str(g["numa_node"]) for g in ranks],
           "--disable-custom-all-reduce", "--language-model-only",
           "--prefill-attention-backend", "triton", "--decode-attention-backend", "trtllm_mha",
           "--linear-attn-prefill-backend", "flashinfer", "--linear-attn-decode-backend", "flashinfer",
           "--fp4-gemm-backend", "flashinfer_cutlass", "--moe-runner-backend", "flashinfer_cutlass",
           "--ple-offload-embedding", "--mamba-ssm-dtype", "bfloat16",
           "--kv-cache-dtype", "fp8_e4m3" if a.kv_dtype == "fp8" else "bfloat16",
           "--mamba-radix-cache-strategy", "extra_buffer", "--max-mamba-cache-size", str(a.mamba_slots),
           "--kt-weight-path", a.gguf, "--kt-method", "LLAMAFILE", "--kt-num-gpu-experts", str(hot),
           "--kt-cpuinfer", str(threads), "--kt-threadpool-count", str(pools), "--kt-max-deferred-experts-per-token", "0",
           "--context-length", str(context), "--max-total-tokens", str(context), "--page-size", "64",
           "--chunked-prefill-size", str(chunk), "--max-prefill-tokens", str(chunk),
           "--max-running-requests", "1", "--cuda-graph-max-bs-decode", "1", "--mem-fraction-static", str(mem_fraction),
           "--disable-flashinfer-autotune", "--sampling-backend", "pytorch", "--reasoning-parser", "qwen3",
           "--watchdog-timeout", "1800", "--warmups", "kt_extend"]
    if a.placement:
        cmd += ["--init-expert-location", a.placement]
    if a.draft_dir:
        cmd += ["--speculative-draft-model-path", a.draft_dir, "--speculative-algorithm", "NEXTN",
                "--speculative-num-steps", "1", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "2"]

    run = os.path.join(a.log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run, exist_ok=True)
    manifest = {"command": cmd, "env": {k: v for k, v in env.items() if k.startswith(("KT_", "SGLANG_", "CUDA_", "OMP_", "NCCL_"))},
                "plan": plan, "host": {k: host[k] for k in ("hostname", "kernel", "driver", "cuda_toolkit")},
                "packages": {}}
    for n in ("torch", "flashinfer-python", "triton", "kt-kernel", "sglang"):
        try:
            manifest["packages"][n] = md.version(n)
        except md.PackageNotFoundError:
            manifest["packages"][n] = None
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"plan: hot {hot} per layer (plan {plan['hot_experts_per_layer']}, lent ~{plan['lent_from_empty_pool']}), "
          f"chunk {chunk}, threads {threads} on nodes {[g['numa_node'] for g in ranks]}, context {context}, mem {mem_fraction}")
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
    if a.background:
        print(f"started pid {proc.pid}")
        return 0
    try:
        while proc.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        proc.terminate()
    return proc.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
