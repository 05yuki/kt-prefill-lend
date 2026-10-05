"""GPU-streamed prefill for KT CPU experts, NVFP4 (ModelOpt) edition.

Above KT_GPU_STREAM_PREFILL tokens a MoE layer's CPU experts are not computed
on the CPU: each rank DMAs its TP part of the experts, G at a time, straight
out of kt-kernel's memfd arenas (KT_EXPERT_SHM=1) into a device slot,
repacks them into the FlashInfer CUTLASS NVFP4 layout on the device, and
runs the same cutlass_fused_moe the resident GPU experts use. Two slots
double-buffer DMA against compute; each (token count, group) is one CUDA
graph so the per-launch host cost under a saturated PCIe link stays off the
critical path (the DSV4.1 lesson, 09-13).

Per-layer fixed cost is the PCIe transfer of the whole expert bank:
Flash-Next is 512 experts x 3 x 2560 x 640 nibbles = 1.26 GB per layer,
split over two cards, ~24 ms at 26 GB/s each.

kt-kernel keeps NVFP4 block scales as fp32 (e4m3 x per-tensor scale_2, folded
via bf16 by its loader); the e4m3 bytes the GPU kernel wants are recovered
exactly by dividing by scale_2 and rounding to e4m3 (the bf16 error of
2^-8 is far inside e4m3's 2^-3 spacing). scale_2 per expert and projection
is read from the checkpoint at init.

Env:
  KT_GPU_STREAM_PREFILL=<tokens>   threshold (0 = off)
  KT_GPU_STREAM_GROUP=<G>          experts per slot (default 8)
  KT_EXPERT_SHM=1                  required (kt-kernel arenas)
  KT_GPU_STREAM_TIMING=1           per-layer timing lines
  KT_GPU_STREAM_SELFTEST=1         rank 0 checks arena bytes and recovered
                                   scales against the checkpoint at init
  KT_GPU_STREAM_DEBUG=<s>          periodic main-thread stack dump
"""
import json
import os
import time
import traceback
from typing import Dict, List, Optional

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

STREAM_THRESHOLD = int(os.environ.get("KT_GPU_STREAM_PREFILL", "0") or 0)
GROUP_SIZE = int(os.environ.get("KT_GPU_STREAM_GROUP", "8") or 8)
TIMING = os.environ.get("KT_GPU_STREAM_TIMING") == "1"
# Chunks shorter than the prefill chunk run padded through the full chunk's
# graphs (patches/sglang-qwen38-stream-fixed-rows.py).
FIXED_ROWS = os.environ.get("KT_GPU_STREAM_FIXED_ROWS", "0") == "1"


def _chunk_rows() -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().chunked_prefill_size or 0)
    except Exception:
        return 0
# One-byte block scales from pinned host memory instead of kt-kernel's FP32
# scale arenas (patches/sglang-qwen38-stream-scale-bytes.py).
SCALE_BYTES = os.environ.get("KT_GPU_STREAM_SCALE_BYTES", "0") == "1"
SCALE_BYTES_VERIFY = os.environ.get("KT_GPU_STREAM_SCALE_BYTES_VERIFY", "0") == "1"


def _log(msg: str):
    print(f"[kt-stream] {msg}", flush=True)


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    """FlashInfer's NVFP4 block-scale layout (sglang swizzle_blockscale), on
    the device, no padding: [n, M, Kc] -> [n, M, Kc] with M % 128 == 0 and
    Kc % 4 == 0."""
    n, M, Kc = s.shape
    return (
        s.reshape(n, M // 128, 4, 32, Kc // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(n, M, Kc)
    )


def _to_e4m3(d: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """fp32 folded scales [n, M, Kc] / scale_2 [n] -> e4m3 bytes."""
    return (d / scale_2.view(-1, 1, 1)).to(torch.float8_e4m3fn).view(torch.uint8)


def _align64(n: int) -> int:
    return (n + 63) & ~63


class _DeviceSlot:
    def __init__(self, G: int, N: int, K: int, device):
        u8 = torch.uint8
        self.w13_weight = torch.empty((G, 2 * N, K // 2), dtype=u8, device=device)
        self.w2_weight = torch.empty((G, K, N // 2), dtype=u8, device=device)
        self.w13_scale = torch.empty((G, 2 * N, K // 16), dtype=u8, device=device)
        self.w2_scale = torch.empty((G, K, N // 16), dtype=u8, device=device)
        self.dma_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.compute_used = False
        # The group's gate/up/down blocks exactly as kt-kernel stores them
        # (packed weights; fp32 scales in separate arenas), G consecutive
        # experts per arena, 64-aligned stride (kt-kernel's arena stride).
        self.raw = torch.empty((3, G, _align64(N * K // 2)), dtype=u8, device=device)
        self.raw_s = torch.empty((3, G, _align64(N * (K // 16) * 4)), dtype=u8, device=device)


# Device buffers made before SGLang sizes its KV pool (preallocate() from
# ModelOptNvFp4FusedMoEMethod.create_moe_runner), so the streamer's ~1 GB
# is not taken from the headroom the pool leaves: with 32 hot experts the
# first 2048-token chunk OOMed when it was the first request (09-14).
_prealloc = {}


def preallocate(layer):
    if STREAM_THRESHOLD <= 0 or not hasattr(layer, "w13_weight"):
        return
    device = layer.w13_weight.device
    key = device.index
    if key in _prealloc:
        return
    N = layer.intermediate_size_per_partition
    K = layer.hidden_size
    slots = [_DeviceSlot(GROUP_SIZE, N, K, device) for _ in range(2)]
    _prealloc[key] = slots
    mb = sum(t.numel() for sl in slots for t in (sl.w13_weight, sl.w2_weight, sl.w13_scale, sl.w2_scale, sl.raw, sl.raw_s)) / 1e6
    _log(f"preallocated 2 device slots ({mb:.0f} MB) on {device} before the KV pool")


class KTStreamPrefill:
    """Process-wide (per rank) streaming state, shared by every MoE layer."""

    def __init__(self, method, layer):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import _activation_type

        self.rank = get_tensor_model_parallel_rank()
        self.world = get_tensor_model_parallel_world_size()
        self.device = layer.w13_weight.device
        self.N = layer.intermediate_size_per_partition
        self.K = layer.hidden_size
        self.G = GROUP_SIZE
        E_res, twoN, Khalf = layer.w13_weight.shape
        if twoN != 2 * self.N or Khalf != self.K // 2:
            raise RuntimeError(
                f"KT_GPU_STREAM_PREFILL: w13 {tuple(layer.w13_weight.shape)} vs N={self.N} K={self.K}")
        bs = layer.w13_blockscale_swizzled
        if tuple(bs.shape[1:]) != (2 * self.N, self.K // 16) or (2 * self.N) % 128 or (self.K // 16) % 4:
            raise RuntimeError(
                f"KT_GPU_STREAM_PREFILL: padded block scales {tuple(bs.shape)} are not supported")
        if self.K % 128 or (self.N // 16) % 4:
            raise RuntimeError("KT_GPU_STREAM_PREFILL: w2 block scales would need padding")
        self.slots = _prealloc.get(self.device.index) or [_DeviceSlot(self.G, self.N, self.K, self.device) for _ in range(2)]
        self._tp_size, self._tp_rank = layer.moe_tp_size, layer.moe_tp_rank
        self._ep_size, self._ep_rank = layer.moe_ep_size, layer.moe_ep_rank
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._cutlass_fused_moe = cutlass_fused_moe
        self._activation_type = _activation_type(layer.moe_runner_config)
        assert self._activation_type == ActivationType.Swiglu, self._activation_type
        self._use_fused_finalize = envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
        self._workspace = None
        self._workspace_tokens = 0
        self._graphs = {}
        self._gin = {}
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._graph_pool = torch.cuda.graph_pool_handle()
        self._arenas = {}
        self._registered = set()
        self._map_expert_arenas(method)
        self._s2 = {}      # layer_idx -> (scale_2 gate/up [E], scale_2 down [E]) on device
        self._alphas = {}  # layer_idx -> (g1 [E], g2 [E]) on device, from the layer's input scales
        self._load_scale_2(method)
        self._sb = {}      # layer_idx -> (w13 scale bytes [E, 2N, K/16], w2 scale bytes [E, K, N/16]), pinned host
        self.calls = 0
        _log(
            f"rank {self.rank}: streamed prefill above {STREAM_THRESHOLD} tokens, G={self.G}, "
            f"N={self.N} K={self.K}, zero-copy from kt-kernel's expert arenas, "
            f"2 device slots x {(self.slots[0].w13_weight.numel() + self.slots[0].w2_weight.numel())/1e6:.0f} MB")
        if self.rank == 0 and os.environ.get("KT_GPU_STREAM_SELFTEST") == "1":
            self._selftest(method)

    # -- checkpoint scale_2 ---------------------------------------------------

    def _load_scale_2(self, method):
        """Per expert and projection scale_2 for every KT layer, from the
        safetensors index; g1/g2 alphas = the layer's input scale x scale_2."""
        from safetensors import safe_open

        path = method.kt_config.weight_path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        # One expert's keys tell us the prefix pattern.
        sample = next(k for k in idx if ".mlp.experts.0.gate_proj.weight_scale_2" in k)
        tpl = sample.split(".layers.")[0] + ".layers.{L}.mlp.experts.{E}.{P}_proj.weight_scale_2"
        files = {}
        t0 = time.time()
        # Every rank (the wrapper and KT_LAYERS exist on rank 0 only): the
        # layers and expert count come from the mapped arenas.
        for layer_idx in sorted(self._arenas):
            E = self._num_experts(layer_idx)
            g = torch.empty(E, dtype=torch.float32)
            d = torch.empty(E, dtype=torch.float32)
            for e in range(E):
                for proj, dst in (("gate", g), ("down", d)):
                    k = tpl.format(L=layer_idx, E=e, P=proj)
                    fn = idx[k]
                    f = files.get(fn)
                    if f is None:
                        f = files[fn] = safe_open(os.path.join(path, fn), "pt")
                    dst[e] = f.get_tensor(k).float().item()
            self._s2[layer_idx] = (g.to(self.device), d.to(self.device))
        _log(f"rank {self.rank}: scale_2 for {len(self._s2)} layers read in {time.time()-t0:.1f} s")

    def _num_experts(self, layer_idx: int) -> int:
        t, stride, _ = self._arenas[layer_idx][0]
        return t.numel() // stride

    def _alphas_for(self, layer, layer_idx: int):
        """g1/g2 = input scale x scale_2 per expert, as the resident method's
        g1_alphas/g2_alphas; the layer keeps 1/input_scale."""
        a = self._alphas.get(layer_idx)
        if a is None:
            s2g, s2d = self._s2[layer_idx]
            g1 = (s2g / layer.w13_input_scale_quant.reshape(-1)[0]).to(torch.float32)
            g2 = (s2d / layer.w2_input_scale_quant.reshape(-1)[0]).to(torch.float32)
            a = self._alphas[layer_idx] = (g1, g2)
        return a

    # -- zero-copy: map kt-kernel's expert arenas -----------------------------

    def _map_expert_arenas(self, method):
        """Rank 0 reads every KT layer's (fd, size, stride) for the part that
        feeds each rank and passes the other ranks their fds over a unix
        socket; every rank mmaps its part."""
        import mmap
        import socket

        from sglang.srt.layers.moe.kt_ep_wrapper import KT_LAYERS

        sock_path = f"/tmp/kt_stream_{os.getppid()}_r{self.rank}.sock"
        infos = {}
        if self.rank == 0:
            per_rank = {r: {} for r in range(self.world)}
            for layer_idx, m in sorted(KT_LAYERS.items()):
                parts = m.wrapper.moe.expert_arena_infos()
                if not parts or len(parts[0]) != 6:
                    raise RuntimeError(
                        "KT_GPU_STREAM_PREFILL needs kt-kernel built with expert arenas and KT_EXPERT_SHM=1 "
                        f"(got {0 if not parts else len(parts[0])} arenas per part, want 6)")
                cpu_tp = len(parts)
                for r in range(self.world):
                    part = r * cpu_tp // self.world if cpu_tp >= self.world else r // (self.world // cpu_tp)
                    per_rank[r][layer_idx] = parts[part]
            infos = per_rank[0]
            for r in range(1, self.world):
                path = f"/tmp/kt_stream_{os.getppid()}_r{r}.sock"
                t0 = time.time()
                while True:
                    try:
                        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        c.connect(path)
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        c.close()
                        if time.time() - t0 > 600:
                            raise
                        time.sleep(0.05)
                layers = sorted(per_rank[r])
                fds = [fd for l in layers for (fd, _, _) in per_rank[r][l]]
                meta = repr([(l, [(sz, st) for (_, sz, st) in per_rank[r][l]]) for l in layers]).encode()
                c.sendall(len(meta).to_bytes(8, "little"))
                c.sendall(meta)
                for i in range(0, len(fds), 200):
                    socket.send_fds(c, [b"x"], fds[i:i + 200])
                c.recv(1)
                c.close()
        else:
            try:
                os.unlink(sock_path)
            except FileNotFoundError:
                pass
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(sock_path)
            srv.listen(1)
            c, _ = srv.accept()
            n = int.from_bytes(c.recv(8), "little")
            buf = b""
            while len(buf) < n:
                buf += c.recv(n - len(buf))
            meta = eval(buf.decode())  # our own repr of ints
            nfds = sum(len(v) for _, v in meta)
            fds = []
            while len(fds) < nfds:
                _, got, _, _ = socket.recv_fds(c, 1, 200)
                fds.extend(got)
            c.sendall(b"k")
            c.close()
            srv.close()
            os.unlink(sock_path)
            i = 0
            for layer_idx, sizes in meta:
                infos[layer_idx] = [(fds[i + j], sizes[j][0], sizes[j][1]) for j in range(6)]
                i += 6
        for layer_idx, arenas in sorted(infos.items()):
            tensors = []
            for (fd, size, stride) in arenas:
                mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
                t = torch.frombuffer(mm, dtype=torch.uint8)
                tensors.append((t, stride, mm))
            self._arenas[layer_idx] = tensors
        _log(f"rank {self.rank}: mapped {len(infos)} layers of expert arenas")

    def _register_arenas(self, layer_idx: int):
        """cudaHostRegister on a layer's first streamed group, so only layers
        that stream get pinned (0.2 s/layer with shmem THP)."""
        if layer_idx in self._registered:
            return
        t0 = time.time()
        total = 0
        for (t, _, _) in self._arenas[layer_idx]:
            rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)
            if int(rc) != 0:
                raise RuntimeError(f"cudaHostRegister(expert arena, layer {layer_idx}) failed: {int(rc)}")
            total += t.numel()
        self._registered.add(layer_idx)
        if TIMING:
            _log(f"rank {self.rank}: registered layer {layer_idx} arenas, {total/1e9:.1f} GB in {time.time()-t0:.1f} s")

    def _selftest(self, method):
        """Rank 0, layer min: arena bytes of this rank's part against the
        checkpoint, and the recovered e4m3 scales against the checkpoint's."""
        from safetensors import safe_open
        from sglang.srt.layers.moe.kt_ep_wrapper import KT_LAYERS

        layer_idx = min(KT_LAYERS)
        m = KT_LAYERS[layer_idx]
        path = method.kt_config.weight_path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        sample = next(k for k in idx if ".mlp.experts.0.gate_proj.weight" in k and k.endswith(".weight"))
        tpl = sample.split(".layers.")[0] + ".layers.{L}.mlp.experts.{E}.{P}_proj.{T}"
        N, K = self.N, self.K
        r = self.rank
        arenas = self._arenas[layer_idx]
        wst, sst = arenas[0][1], arenas[3][1]
        s2g, s2d = self._s2[layer_idx]

        def get(e, proj, t):
            k = tpl.format(L=layer_idx, E=e, P=proj, T=t)
            with safe_open(os.path.join(path, idx[k]), "pt") as f:
                return f.get_tensor(k)

        for e in (m.num_gpu_experts, m.num_gpu_experts + 93, m.global_num_experts - 1):
            ck_gate = get(e, "gate", "weight")[r * N:(r + 1) * N]          # [N, K/2]
            ck_up = get(e, "up", "weight")[r * N:(r + 1) * N]
            ck_down = get(e, "down", "weight")[:, r * (N // 2):(r + 1) * (N // 2)].contiguous()  # [K, N/2]
            ck_gs = get(e, "gate", "weight_scale")[r * N:(r + 1) * N].view(torch.uint8)
            ck_us = get(e, "up", "weight_scale")[r * N:(r + 1) * N].view(torch.uint8)
            ck_ds = get(e, "down", "weight_scale")[:, r * (N // 16):(r + 1) * (N // 16)].contiguous().view(torch.uint8)
            a_gate = arenas[0][0][e * wst:e * wst + N * K // 2].view(N, K // 2)
            a_up = arenas[1][0][e * wst:e * wst + N * K // 2].view(N, K // 2)
            a_down = arenas[2][0][e * wst:e * wst + K * N // 2].view(K, N // 2)
            d_gs = arenas[3][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32).view(1, N, K // 16)
            d_us = arenas[4][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32).view(1, N, K // 16)
            d_ds = arenas[5][0][e * sst:e * sst + 4 * K * (N // 16)].view(torch.float32).view(1, K, N // 16)
            r_gs = _to_e4m3(d_gs, s2g[e:e + 1].cpu())[0]
            r_us = _to_e4m3(d_us, s2g[e:e + 1].cpu())[0]
            r_ds = _to_e4m3(d_ds, s2d[e:e + 1].cpu())[0]

            def mism(a, b):
                return int((a != b).sum())

            _log(f"selftest layer {layer_idx} expert {e} part {r}: weight mismatches gate={mism(a_gate, ck_gate)}/{a_gate.numel()} "
                 f"up={mism(a_up, ck_up)} down={mism(a_down, ck_down)}/{a_down.numel()}; recovered e4m3 scale mismatches "
                 f"gate={mism(r_gs, ck_gs)}/{r_gs.numel()} up={mism(r_us, ck_us)} down={mism(r_ds, ck_ds)}/{r_ds.numel()}; "
                 f"gate_w[:6] arena={a_gate.flatten()[:6].tolist()} ckpt={ck_gate.flatten()[:6].tolist()}")

    # -- staging ---------------------------------------------------------------

    def _stage(self, layer_idx: int, gi: int, experts, stats):
        """Six contiguous H2D copies: the group's consecutive expert blocks of
        the gate, up and down weight and scale arenas into the slot's raw
        landing, on the transfer stream."""
        self._register_arenas(layer_idx)
        slot = self.slots[gi % 2]
        n = len(experts)
        base = experts[0]
        arenas = self._arenas[layer_idx]
        wstride = slot.raw.shape[2]
        sstride = slot.raw_s.shape[2]
        assert all(a[1] == wstride for a in arenas[:3]) and all(a[1] == sstride for a in arenas[3:]),             f"arena strides {[a[1] for a in arenas]} vs slot {wstride}/{sstride}"
        t0 = time.perf_counter()
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            for pi in range(3):
                slot.raw[pi, :n].view(-1).copy_(arenas[pi][0][base * wstride:(base + n) * wstride], non_blocking=True)
                if not SCALE_BYTES:
                    slot.raw_s[pi, :n].view(-1).copy_(
                        arenas[3 + pi][0][base * sstride:(base + n) * sstride], non_blocking=True)
            if SCALE_BYTES:
                h13, h2 = self._sb[layer_idx]
                slot.w13_scale[:n].copy_(h13[base:base + n], non_blocking=True)
                slot.w2_scale[:n].copy_(h2[base:base + n], non_blocking=True)
            slot.dma_done.record(self.transfer_stream)
        stats["dma_enq"] += time.perf_counter() - t0

    def _scales_from_fp32(self, raw_s, n: int, s2g, s2d):
        """kt-kernel's FP32 scale blocks [3, n, stride] -> the swizzled e4m3
        bytes FlashInfer takes: w13 [n, 2N, K/16] in [up; gate] rows, w2
        [n, K, N/16]."""
        N, K = self.N, self.K
        w13_s = N * (K // 16)
        w2_s = K * (N // 16)
        s13 = torch.empty((n, 2 * N, K // 16), dtype=torch.uint8, device=self.device)
        s13[:, :N] = _to_e4m3(raw_s[1, :, :4 * w13_s].view(torch.float32).view(n, N, K // 16), s2g)
        s13[:, N:] = _to_e4m3(raw_s[0, :, :4 * w13_s].view(torch.float32).view(n, N, K // 16), s2g)
        s2 = _to_e4m3(raw_s[2, :, :4 * w2_s].view(torch.float32).view(n, K, N // 16), s2d)
        return _swizzle(s13), _swizzle(s2)

    def _build_scale_bytes(self, layer_idx: int):
        """Run the FP32 -> e4m3 conversion once for every expert of the layer
        and keep the bytes in pinned host memory."""
        if layer_idx in self._sb:
            return
        self._register_arenas(layer_idx)
        N, K = self.N, self.K
        E = self._num_experts(layer_idx)
        arenas = self._arenas[layer_idx]
        slot = self.slots[0]
        sstride = slot.raw_s.shape[2]
        s2g_all, s2d_all = self._s2[layer_idx]
        h13 = torch.empty((E, 2 * N, K // 16), dtype=torch.uint8).pin_memory()
        h2 = torch.empty((E, K, N // 16), dtype=torch.uint8).pin_memory()
        cur = torch.cuda.current_stream(self.device)
        if slot.compute_used:
            cur.wait_event(slot.compute_done)
        cur.wait_stream(self.transfer_stream)
        for base in range(0, E, self.G):
            n = min(self.G, E - base)
            for pi in range(3):
                slot.raw_s[pi, :n].view(-1).copy_(arenas[3 + pi][0][base * sstride:(base + n) * sstride])
            s13, s2 = self._scales_from_fp32(slot.raw_s[:, :n], n, s2g_all[base:base + n], s2d_all[base:base + n])
            h13[base:base + n].copy_(s13)
            h2[base:base + n].copy_(s2)
        self._sb[layer_idx] = (h13, h2)

    def _verify_scale_bytes(self, layer_idx: int, gi: int, base: int, n: int):
        """KT_GPU_STREAM_SCALE_BYTES_VERIFY: the bytes the DMA put in the slot
        against the FP32 arenas converted now. Synchronous; a diagnostic."""
        slot = self.slots[gi % 2]
        arenas = self._arenas[layer_idx]
        sstride = slot.raw_s.shape[2]
        torch.cuda.current_stream(self.device).wait_event(slot.dma_done)
        for pi in range(3):
            slot.raw_s[pi, :n].view(-1).copy_(arenas[3 + pi][0][base * sstride:(base + n) * sstride])
        s2g_all, s2d_all = self._s2[layer_idx]
        s13, s2 = self._scales_from_fp32(slot.raw_s[:, :n], n, s2g_all[base:base + n], s2d_all[base:base + n])
        bad = int((s13 != slot.w13_scale[:n]).sum().item()) + int((s2 != slot.w2_scale[:n]).sum().item())
        self._verified = getattr(self, "_verified", 0) + 1
        if bad:
            raise RuntimeError(f"KT_GPU_STREAM_SCALE_BYTES: layer {layer_idx} group {gi}: {bad} scale bytes differ")
        if self._verified % 384 == 0:
            _log(f"rank{self.rank} scale bytes verified against the FP32 arenas: {self._verified} groups, 0 mismatches")

    # -- one group ---------------------------------------------------------------

    def _group_body(self, layer, gi: int, base: int, n: int, g):
        """Repack + MoE + accumulate for one group on the current stream;
        captured into a graph per (token count, group). Everything that
        differs per layer (scale_2, alphas) is a graph input in g."""
        slot = self.slots[gi % 2]
        N, K = self.N, self.K
        w13_bytes = N * K // 2
        w13_s = N * (K // 16)
        w2_bytes = K * N // 2
        w2_s = K * (N // 16)
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        s2g, s2d, g1, g2 = g["s2g"][:n], g["s2d"][:n], g["g1"][:n], g["g2"][:n]
        raw = slot.raw[:, :n]
        raw_s = slot.raw_s[:, :n]
        # FlashInfer's CUTLASS MoE takes fc1 as [up; gate] rows (w31), the
        # order SGLang's swap_w13_to_w31 produces for it; the checkpoint's
        # [gate; up] gave outputs with an error above the signal (09-14
        # refcheck on the bf16 edition). Same nibble packing as kt-kernel.
        slot.w13_weight[:n, :N].copy_(raw[1, :, :w13_bytes].view(n, N, K // 2))
        slot.w13_weight[:n, N:].copy_(raw[0, :, :w13_bytes].view(n, N, K // 2))
        slot.w2_weight[:n].copy_(raw[2, :, :w2_bytes].view(n, K, N // 2))
        if not SCALE_BYTES:
            s13, s2 = self._scales_from_fp32(raw_s, n, s2g, s2d)
            slot.w13_scale[:n].copy_(s13)
            slot.w2_scale[:n].copy_(s2)
        # else: _stage put the finished bytes in slot.w13_scale / w2_scale
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=slot.w13_weight[:n].view(torch.int64),
            fc2_expert_weights=slot.w2_weight[:n].view(torch.int64),
            output_dtype=torch.bfloat16,
            quant_scales=[
                g["a1"],
                slot.w13_scale[:n].view(torch.int32),
                g1,
                g["a2"],
                slot.w2_scale[:n].view(torch.int32),
                g2,
            ],
            input_sf=None,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            ep_size=self._ep_size,
            ep_rank=self._ep_rank,
            activation_type=self._activation_type,
            tune_max_num_tokens=self._tune_tokens,
            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
        acc.add_(out)

    def _reference_check(self, method, layer, layer_idx, x, ids, w, acc):
        """This rank's partial MoE output for the streamed experts from a
        torch dequantisation of the arenas (e2m1 nibbles x e4m3 block scale
        x scale_2, bf16 activations, i.e. W4A16), against acc (W4A4 on the
        GPU). Not bit-exact: the activation quantisation is the difference."""
        torch.cuda.current_stream(self.device).synchronize()
        N, K = self.N, self.K
        ar = self._arenas[layer_idx]
        wst, sst = ar[0][1], ar[3][1]
        first = method.num_gpu_experts
        s2g, s2d = self._s2[layer_idx]
        e2m1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], device=self.device)

        def deq(wb, sb, rows, cols, s2):
            q = wb.to(self.device).view(rows, cols // 2)
            lo, hi = (q & 0xF).long(), (q >> 4).long()
            v = torch.stack([e2m1[lo], e2m1[hi]], -1).view(rows, cols)
            sc = sb.to(self.device).view(rows, cols // 16).float().repeat_interleave(16, 1)
            return v * sc * s2

        ref = torch.zeros_like(acc, dtype=torch.float32)
        ids64 = ids.to(torch.int64)
        xf = x.float()
        used = 0
        for e in range(first, self._num_experts(layer_idx)):
            rows, cols = torch.where(ids64 == e)
            if rows.numel() == 0:
                continue
            used += 1
            gs = ar[3][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32)
            us = ar[4][0][e * sst:e * sst + 4 * N * (K // 16)].view(torch.float32)
            ds = ar[5][0][e * sst:e * sst + 4 * K * (N // 16)].view(torch.float32)
            gate = deq(ar[0][0][e * wst:e * wst + N * K // 2], gs, N, K, 1.0)  # d already = e4m3 x scale_2
            up = deq(ar[1][0][e * wst:e * wst + N * K // 2], us, N, K, 1.0)
            down = deq(ar[2][0][e * wst:e * wst + K * N // 2], ds, K, N, 1.0)
            h = xf[rows]
            a = torch.nn.functional.silu(h @ gate.t()) * (h @ up.t())
            o = (a @ down.t()) * w[rows, cols].float().unsqueeze(1)
            ref.index_add_(0, rows, o)
        d = (acc.float() - ref).abs()
        cos = torch.nn.functional.cosine_similarity(acc.float().flatten(), ref.flatten(), dim=0).item()
        _log(f"rank{self.rank} refcheck layer{layer_idx} T={x.shape[0]} experts_used={used}: cos={cos:.5f} "
             f"max|acc-ref|={d.max().item():.4f} mean={d.mean().item():.5f} ref_mean|.|={ref.abs().mean().item():.5f} "
             f"acc_mean|.|={acc.float().abs().mean().item():.5f}")

    def _graph_inputs(self, T: int, x, physical_ids, topk_weights):
        """Persistent graph inputs for this token count; the per-layer
        constants (input scales, scale_2 and alphas of the current group)
        are copied in before each replay."""
        g = self._gin.get(T)
        n = x.shape[0]
        if g is None:
            f32 = torch.float32
            g = {
                "x": torch.zeros((T, x.shape[1]), dtype=x.dtype, device=x.device),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=f32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
                "a1": torch.ones((), dtype=f32, device=x.device),  # scalar: FlashInfer checks ndim
                "a2": torch.ones((), dtype=f32, device=x.device),
                "s2g": torch.ones((self.G,), dtype=f32, device=x.device),
                "s2d": torch.ones((self.G,), dtype=f32, device=x.device),
                "g1": torch.ones((self.G,), dtype=f32, device=x.device),
                "g2": torch.ones((self.G,), dtype=f32, device=x.device),
            }
            self._gin[T] = g
        g["x"][:n].copy_(x)
        g["ids"][:n].copy_(physical_ids)
        g["w"][:n].copy_(topk_weights)
        if n < T:
            # padding rows: no expert in any group, nothing added to acc
            g["x"][n:].zero_()
            g["ids"][n:].fill_(-1)
            g["w"][n:].zero_()
        g["acc"].zero_()
        return g

    def _set_layer_inputs(self, g, layer):
        g["a1"].copy_(layer.w13_input_scale_quant.reshape(-1)[0])
        g["a2"].copy_(layer.w2_input_scale_quant.reshape(-1)[0])

    def _set_group_inputs(self, g, layer, layer_idx: int, base: int, n: int):
        s2g, s2d = self._s2[layer_idx]
        g1, g2 = self._alphas_for(layer, layer_idx)
        g["s2g"][:n].copy_(s2g[base:base + n])
        g["s2d"][:n].copy_(s2d[base:base + n])
        g["g1"][:n].copy_(g1[base:base + n])
        g["g2"][:n].copy_(g2[base:base + n])

    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
        key = (T, gi)
        graph = self._graphs.get(key)
        if graph is None:
            t0 = time.perf_counter()
            cs = self._capture_stream
            cs.wait_stream(torch.cuda.current_stream(self.device))
            # The eager warm-up (lazy inits, kernel selection) must not touch
            # the real accumulator: it ran acc.add_ once more than the replay
            # and the first request per chunk size carried the streamed
            # experts twice (09-14 refcheck).
            gw = dict(g)
            gw["acc"] = torch.zeros_like(g["acc"])
            with torch.cuda.stream(cs):
                self._group_body(layer, gi, base, n, gw)
            cs.synchronize()
            del gw
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool, stream=cs):
                self._group_body(layer, gi, base, n, g)
            torch.cuda.current_stream(self.device).wait_stream(cs)
            self._graphs[key] = graph
            if TIMING and gi == 0:
                _log(f"rank{self.rank} captured group graph T={T} gi={gi} n={n} in {(time.perf_counter()-t0)*1e3:.0f} ms")
        return graph

    # -- one layer ----------------------------------------------------------

    def run(self, method, layer, dispatch_output, physical_ids: torch.Tensor) -> torch.Tensor:
        """Sum of the streamed experts' outputs for this layer, [tokens, hidden]."""
        from sglang.srt.utils.common import next_power_of_2

        x = dispatch_output.hidden_states
        rows = x.shape[0]
        T = rows
        if FIXED_ROWS and rows < _chunk_rows():
            T = _chunk_rows()
        layer_idx = method.kt_config.layer_idx
        self._tune_tokens = next_power_of_2(T)
        if self._workspace is None or self._workspace_tokens < T:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                fp4_workspace_nbytes,
                get_shared_cutlass_workspace,
            )

            nbytes = fp4_workspace_nbytes(
                T, self.K, self.N, self.G * layer.moe_ep_size, physical_ids.shape[1],
                self._activation_type, layer.moe_tp_size, layer.moe_tp_rank,
                layer.moe_ep_size, layer.moe_ep_rank, x.device)
            self._workspace = get_shared_cutlass_workspace(nbytes, x.device)
            if self._workspace is None:
                raise RuntimeError(
                    "KT streamed prefill: the shared CUTLASS workspace was not preallocated "
                    "large enough (set KT_GPU_STREAM_GROUP before launch)")
            self._workspace_tokens = T
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {T} tokens x {self.G} experts")
        topk_weights = dispatch_output.topk_output.topk_weights
        first = method.num_gpu_experts
        last = self._num_experts(layer_idx)
        groups = [list(range(b, min(b + self.G, last))) for b in range(first, last, self.G)]
        stats = {"dma_enq": 0.0, "graph_enq": 0.0}
        t0 = time.perf_counter()
        g = self._graph_inputs(T, x, physical_ids, topk_weights)
        self._set_layer_inputs(g, layer)
        cur = torch.cuda.current_stream(self.device)
        if SCALE_BYTES:
            self._build_scale_bytes(layer_idx)
        self._stage(layer_idx, 0, groups[0], stats)
        for gi, experts in enumerate(groups):
            slot = self.slots[gi % 2]
            graph = self._graph_for(layer, T, gi, experts[0], len(experts), g)
            t1 = time.perf_counter()
            self._set_group_inputs(g, layer, layer_idx, experts[0], len(experts))
            cur.wait_event(slot.dma_done)
            if SCALE_BYTES_VERIFY and SCALE_BYTES:
                self._verify_scale_bytes(layer_idx, gi, experts[0], len(experts))
            graph.replay()
            slot.compute_done.record(cur)
            slot.compute_used = True
            stats["graph_enq"] += time.perf_counter() - t1
            if gi + 1 < len(groups):
                self._stage(layer_idx, gi + 1, groups[gi + 1], stats)
        self.calls += 1
        if TIMING:
            cur.synchronize()
            _log(f"rank{self.rank} layer{layer_idx} tokens={rows}/{T} groups={len(groups)} "
                 f"dma_enq={stats['dma_enq']*1e3:.0f}ms graph_enq={stats['graph_enq']*1e3:.0f}ms "
                 f"total={(time.perf_counter()-t0)*1e3:.0f}ms")
        if os.environ.get("KT_GPU_STREAM_SELFTEST") == "2" and layer_idx == min(self._arenas):
            self._reference_check(method, layer, layer_idx, x, physical_ids, topk_weights, g["acc"][:rows])
        # The caller adds this before the next layer zeroes it (same stream).
        return g["acc"] if rows == T else g["acc"][:rows]


_instance: Optional[KTStreamPrefill] = None
_failed = False


def _start_stack_dumper():
    period = float(os.environ.get("KT_GPU_STREAM_DEBUG", "0") or 0)
    if period <= 0:
        return
    import sys
    import threading

    main_id = threading.main_thread().ident
    rank = get_tensor_model_parallel_rank()

    def loop():
        while True:
            time.sleep(period)
            fr = sys._current_frames().get(main_id)
            if fr is not None:
                _log(f"rank{rank} main thread stack:" + chr(10) + "".join(traceback.format_stack(fr)[-12:]))

    threading.Thread(target=loop, daemon=True, name="kt-stream-stackdump").start()


# KT_GPU_STREAM_EARLY_INIT=1: build the streamer and cudaHostRegister every
# layer's arenas right after the weights load, instead of on the first
# qualifying request (09-26: 15.5 min on the first 38K prompt, 4K pages).
EARLY_INIT = os.environ.get("KT_GPU_STREAM_EARLY_INIT", "0") == "1"


def stream_prefill_init_early(method, layer) -> None:
    global _instance, _failed
    if not EARLY_INIT or STREAM_THRESHOLD <= 0 or _failed or _instance is not None:
        return
    rank = get_tensor_model_parallel_rank()
    t0 = time.time()
    try:
        _instance = KTStreamPrefill(method, layer)
        _start_stack_dumper()
        for layer_idx in sorted(_instance._arenas):
            _instance._register_arenas(layer_idx)
        if SCALE_BYTES:
            t1 = time.time()
            for layer_idx in sorted(_instance._arenas):
                _instance._build_scale_bytes(layer_idx)
            nb = sum(a.numel() + b.numel() for a, b in _instance._sb.values())
            _log(f"rank {rank}: one-byte block scales for {len(_instance._sb)} layers, "
                 f"{nb/1e9:.2f} GB pinned, built in {time.time() - t1:.1f} s")
    except Exception:
        _failed = True
        _log("early init failed:\n" + traceback.format_exc())
        raise
    _log(f"rank {rank}: streamer built and {len(_instance._arenas)} layers of arenas registered "
         f"at load time in {time.time() - t0:.0f} s")


def stream_prefill_for(method, layer, num_tokens: int) -> Optional[KTStreamPrefill]:
    """The per-rank streamer when this call qualifies, else None."""
    global _instance, _failed
    if STREAM_THRESHOLD <= 0 or _failed:
        return None
    if num_tokens < STREAM_THRESHOLD or torch.cuda.is_current_stream_capturing():
        return None
    if _instance is None:
        _log(f"rank {get_tensor_model_parallel_rank()}: first qualifying call, layer {method.kt_config.layer_idx}, tokens={num_tokens}")
        try:
            _instance = KTStreamPrefill(method, layer)
            _start_stack_dumper()
        except Exception:
            _failed = True
            _log("init failed:\n" + traceback.format_exc())
            raise
    return _instance
