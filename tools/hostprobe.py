#!/usr/bin/env python3
"""The host as a launcher sees it: GPUs with their NUMA node, VRAM and SM count; NUMA nodes with their CPUs and
RAM; CPU ISA flags; the memory settings the launchers depend on (THP, swap, min_free_kbytes); driver / CUDA /
kernel; the model mounts; what is using the GPUs now. JSON on stdout (--json) or a table. No host-specific values
inside: every launcher that still carries a GPU UUID or a CPU count should read this instead.

usage: hostprobe.py [--json] [--gpu-on-node N] [--cpus-of-node N] [--mounts PATH ...]
  --gpu-on-node N   print the UUIDs (comma-separated) of the GPUs on NUMA node N, nothing else
  --cpus-of-node N  print the number of physical cores on NUMA node N (threads to give a CPU expert pool)
Reads /sys, /proc and nvidia-smi only; needs no venv.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys


def _read(p, default=""):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def _cpulist(s):
    out = []
    for part in s.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def numa_nodes():
    nodes = []
    base = "/sys/devices/system/node"
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if not re.fullmatch(r"node\d+", d):
            continue
        n = int(d[4:])
        cpus = _cpulist(_read(f"{base}/{d}/cpulist"))
        # physical cores: one per distinct (core id) among this node's CPUs
        cores = set()
        for c in cpus:
            cores.add(_read(f"/sys/devices/system/cpu/cpu{c}/topology/core_id", str(c)))
        mem = {}
        for line in _read(f"{base}/{d}/meminfo").splitlines():
            m = re.match(r"Node \d+ (\w+):\s+(\d+) kB", line)
            if m:
                mem[m.group(1)] = int(m.group(2)) * 1024
        nodes.append({"node": n, "cpus": cpus, "physical_cores": len(cores), "mem_total": mem.get("MemTotal"),
                      "mem_free": mem.get("MemFree"), "file_pages": mem.get("FilePages"),
                      "anon_huge": mem.get("AnonHugePages")})
    return nodes


def cpu_info():
    text = _read("/proc/cpuinfo")
    flags = set()
    model = ""
    for line in text.splitlines():
        if line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
        elif line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
    isa = {k: (k in flags) for k in ("avx2", "avx512f", "avx512_bf16", "avx512_vnni", "amx_bf16", "amx_int8", "fma")}
    return {"model": model, "isa": isa, "logical_cpus": os.cpu_count()}


def gpus():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return [], {}
    q = "index,uuid,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap"
    out = subprocess.run([smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = []
    driver = ""
    for line in out.strip().splitlines():
        idx, uuid, bus, name, total, used, driver, cc = [v.strip() for v in line.split(",")]
        bus_l = bus.lower()
        if bus_l.startswith("00000000:"):
            bus_l = bus_l[4:]  # /sys uses the 4-digit domain
        node = _read(f"/sys/bus/pci/devices/{bus_l}/numa_node", "-1")
        res.append({"index": int(idx), "uuid": uuid, "bus": bus, "numa_node": int(node), "name": name,
                    "vram_total": int(total) * 2**20, "vram_used": int(used) * 2**20, "compute_cap": cc})
    procs = subprocess.run([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    using = []
    for line in procs.strip().splitlines():
        u, pid, pname, mem = [v.strip() for v in line.split(",")]
        using.append({"gpu_uuid": u, "pid": int(pid), "name": pname, "used": int(mem) * 2**20})
    cuda = ""
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        v = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release (\d+\.\d+)", v)
        cuda = m.group(1) if m else ""
    return res, {"driver": driver, "cuda_toolkit": cuda, "in_use": using}


def memory_settings():
    swap = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("SwapTotal"):
            swap = int(line.split()[1]) * 1024
    pick = lambda s: (re.search(r"\[(\w+)\]", s) or [None, s])[1]
    return {"thp_enabled": pick(_read("/sys/kernel/mm/transparent_hugepage/enabled")),
            "thp_shmem": pick(_read("/sys/kernel/mm/transparent_hugepage/shmem_enabled")),
            "thp_defrag": pick(_read("/sys/kernel/mm/transparent_hugepage/defrag")),
            "swap_total": swap, "min_free_kbytes": int(_read("/proc/sys/vm/min_free_kbytes", "0") or 0),
            "overcommit_memory": int(_read("/proc/sys/vm/overcommit_memory", "0") or 0),
            "hugepages_2m": int(_read("/proc/sys/vm/nr_hugepages", "0") or 0)}


def mounts(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            out.append({"path": p, "exists": False})
            continue
        st = os.statvfs(p)
        src = ""
        for line in _read("/proc/mounts").splitlines():
            dev, mnt, fstype = line.split()[:3]
            if mnt != "/" and (p == mnt or p.startswith(mnt + "/")) and len(mnt) > len(src):
                src, fs = mnt, fstype
        out.append({"path": p, "exists": True, "free": st.f_bavail * st.f_frsize, "total": st.f_blocks * st.f_frsize,
                    "mount": src or "/"})
    return out


def probe(mount_paths):
    g, gmeta = gpus()
    return {"hostname": os.uname().nodename, "kernel": os.uname().release, "cpu": cpu_info(), "numa": numa_nodes(),
            "gpus": g, "driver": gmeta.get("driver"), "cuda_toolkit": gmeta.get("cuda_toolkit"),
            "gpu_in_use": gmeta.get("in_use", []), "memory": memory_settings(),
            "load_1m": os.getloadavg()[0], "mounts": mounts(mount_paths)}


def table(h):
    gb = lambda b: f"{(b or 0) / 2**30:.1f} GiB"
    lines = [f"{h['hostname']}  kernel {h['kernel']}  driver {h['driver']}  cuda {h['cuda_toolkit']}  load {h['load_1m']:.1f}",
             f"cpu: {h['cpu']['model']}  logical {h['cpu']['logical_cpus']}  isa " +
             " ".join(k for k, v in h["cpu"]["isa"].items() if v)]
    for n in h["numa"]:
        c = n["cpus"]
        lines.append(f"node {n['node']}: {n['physical_cores']} cores ({len(c)} cpus {c[0]}-{c[-1]}...)  RAM {gb(n['mem_total'])}, "
                     f"free {gb(n['mem_free'])}, page cache {gb(n['file_pages'])}")
    for g in h["gpus"]:
        used = [u for u in h["gpu_in_use"] if u["gpu_uuid"] == g["uuid"]]
        lines.append(f"gpu {g['index']}: {g['name']} cc {g['compute_cap']} bus {g['bus']} node {g['numa_node']} "
                     f"VRAM {gb(g['vram_total'])} used {gb(g['vram_used'])}  {g['uuid']}" +
                     ("  in use by " + ", ".join(f"{u['name']}({u['pid']})" for u in used) if used else ""))
    m = h["memory"]
    lines.append(f"memory: THP {m['thp_enabled']} / shmem {m['thp_shmem']} / defrag {m['thp_defrag']}  swap {gb(m['swap_total'])}  "
                 f"min_free_kbytes {m['min_free_kbytes'] // 1024} MiB  overcommit {m['overcommit_memory']}")
    for mt in h["mounts"]:
        lines.append(f"{mt['path']}: " + (f"free {gb(mt['free'])} of {gb(mt['total'])} ({mt['mount']})" if mt["exists"] else "missing"))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--gpu-on-node", type=int)
    ap.add_argument("--cpus-of-node", type=int)
    ap.add_argument("--mounts", nargs="*", default=[os.getcwd()], help="model directories to report free space for")
    a = ap.parse_args()
    h = probe(a.mounts)
    if a.gpu_on_node is not None:
        print(",".join(g["uuid"] for g in h["gpus"] if g["numa_node"] == a.gpu_on_node))
        return
    if a.cpus_of_node is not None:
        n = [x for x in h["numa"] if x["node"] == a.cpus_of_node]
        print(n[0]["physical_cores"] if n else 0)
        return
    print(json.dumps(h, indent=1) if a.json else table(h))


if __name__ == "__main__":
    sys.exit(main())
