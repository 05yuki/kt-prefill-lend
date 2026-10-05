"""GPU-streamed prefill for KT CPU experts, BF16 edition (Qwen3.5-35B-A3B).

Above KT_GPU_STREAM_PREFILL tokens a MoE layer's CPU experts are not computed
on the CPU: each rank DMAs its TP part of the experts, G at a time, straight
out of kt-kernel's memfd arenas (KT_EXPERT_SHM=1) into a device slot,
repacks them into the FlashInfer CUTLASS NVFP4 layout on the device, and
runs the same cutlass_fused_moe the resident GPU experts use. Two slots
double-buffer DMA against compute; each (token count, group) is one CUDA
graph so the per-launch host cost under a saturated PCIe link stays off the
critical path (the DSV4.1 lesson, 09-13).

Per-layer fixed cost is the PCIe transfer of the whole expert bank:
256 experts x 3 x 2048 x 512 x 2 B = 1.6 GB per layer, split over two
cards, ~30 ms at 26 GB/s each.

kt-kernel's BF16 BufferB is the plain row-major weight ([N, K] for gate/up,
[K, N] for down), which is also FlashInfer's bf16 MoE layout, so the
"repack" is three copies into the [up; gate] / down slot tensors and the
MoE runs bf16 x bf16, the same arithmetic class as the CPU path.

Env:
  KT_GPU_STREAM_PREFILL=<tokens>   threshold (0 = off)
  KT_GPU_STREAM_GROUP=<G>          experts per slot (default 8)
  KT_EXPERT_SHM=1                  required (kt-kernel arenas)
  KT_GPU_STREAM_TIMING=1           per-layer timing lines
  KT_GPU_STREAM_SELFTEST=1         rank 0 checks arena bytes against the checkpoint
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
USE_GRAPH = os.environ.get("KT_GPU_STREAM_GRAPH", "1") == "1"


def _log(msg: str):
    print(f"[kt-stream] {msg}", flush=True)


class _DeviceSlot:
    def __init__(self, G: int, N: int, K: int, device):
        bf = torch.bfloat16
        self.w13_weight = torch.empty((G, 2 * N, K), dtype=bf, device=device)
        self.w2_weight = torch.empty((G, K, N), dtype=bf, device=device)
        self.dma_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.compute_used = False
        # The group's gate/up/down blocks exactly as kt-kernel stores them,
        # G consecutive experts per arena, 64-aligned stride.
        self.raw = None


class KTStreamPrefill:
    """Process-wide (per rank) streaming state, shared by every MoE layer."""

    def __init__(self, method, layer):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from sglang.srt.environ import envs

        self.rank = get_tensor_model_parallel_rank()
        self.world = get_tensor_model_parallel_world_size()
        self.device = layer.w13_weight.device
        self.N = layer.intermediate_size_per_partition
        self.K = layer.hidden_size
        self.G = GROUP_SIZE
        E_res, twoN, Kw = layer.w13_weight.shape
        if layer.w13_weight.dtype != torch.bfloat16 or twoN != 2 * self.N or Kw != self.K:
            raise RuntimeError(
                f"KT_GPU_STREAM_PREFILL (bf16): w13 {tuple(layer.w13_weight.shape)} {layer.w13_weight.dtype} "
                f"vs N={self.N} K={self.K}")
        self.slots = [_DeviceSlot(self.G, self.N, self.K, self.device) for _ in range(2)]
        self._tp_size, self._tp_rank = layer.moe_tp_size, layer.moe_tp_rank
        self._ep_size, self._ep_rank = layer.moe_ep_size, layer.moe_ep_rank
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._cutlass_fused_moe = cutlass_fused_moe
        assert layer.moe_runner_config.activation == "silu", layer.moe_runner_config.activation
        self._activation_type = ActivationType.Swiglu
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
        self.calls = 0
        _log(
            f"rank {self.rank}: streamed prefill above {STREAM_THRESHOLD} tokens, G={self.G}, "
            f"N={self.N} K={self.K}, zero-copy from kt-kernel's expert arenas, "
            f"2 device slots x {(self.slots[0].w13_weight.numel() + self.slots[0].w2_weight.numel())/1e6:.0f} MB")
        if self.rank == 0 and os.environ.get("KT_GPU_STREAM_SELFTEST") == "1":
            self._selftest_layer = layer
            self._selftest(method)

    def _num_experts(self, layer_idx: int) -> int:
        t, stride, _ = self._arenas[layer_idx][0]
        return t.numel() // stride

    def _num_experts(self, layer_idx: int) -> int:
        t, stride, _ = self._arenas[layer_idx][0]
        return t.numel() // stride

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
                if not parts or len(parts[0]) != 3:
                    raise RuntimeError(
                        "KT_GPU_STREAM_PREFILL needs kt-kernel built with expert arenas and KT_EXPERT_SHM=1 "
                        f"(got {0 if not parts else len(parts[0])} arenas per part, want 3)")
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
                infos[layer_idx] = [(fds[i + j], sizes[j][0], sizes[j][1]) for j in range(3)]
                i += 3
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
        """Rank 0, layer min: this rank's part of a few experts' arena bytes
        against the checkpoint. Qwen3.5-MoE stores the experts fused:
        experts.gate_up_proj [E, 2I, H] ([gate; up] rows) and
        experts.down_proj [E, H, I]; the MTP layer keeps per-expert keys."""
        from safetensors import safe_open

        layer_idx = min(self._arenas)
        m = method
        path = method.kt_config.weight_path
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        key_gu = next(k for k in idx if f".layers.{layer_idx}.mlp.experts.gate_up_proj" in k and not k.startswith("mtp"))
        key_d = key_gu.replace("gate_up_proj", "down_proj")
        N, K, r = self.N, self.K, self.rank
        I = N * self.world  # full intermediate size
        arenas = self._arenas[layer_idx]
        wst = arenas[0][1]
        p2l = self._physical_to_logical(layer_idx)
        with safe_open(os.path.join(path, idx[key_gu]), "pt") as f:
            gu = f.get_slice(key_gu)
        with safe_open(os.path.join(path, idx[key_d]), "pt") as f:
            dn = f.get_slice(key_d)
        for e in (m.num_gpu_experts, m.num_gpu_experts + 93, self._num_experts(layer_idx) - 1):
            le = int(p2l[e]) if p2l is not None else e
            ck_gate = gu[le, r * N:(r + 1) * N]
            ck_up = gu[le, I + r * N:I + (r + 1) * N]
            ck_down = dn[le, :, r * N:(r + 1) * N].contiguous()
            a_gate = arenas[0][0][e * wst:e * wst + N * K * 2].view(torch.bfloat16).view(N, K)
            a_up = arenas[1][0][e * wst:e * wst + N * K * 2].view(torch.bfloat16).view(N, K)
            a_down = arenas[2][0][e * wst:e * wst + K * N * 2].view(torch.bfloat16).view(K, N)

            def mism(a, b):
                return int((a != b).sum())

            _log(f"selftest layer {layer_idx} slot {e} (logical {le}) part {r}: mismatches gate={mism(a_gate, ck_gate)}/{a_gate.numel()} "
                 f"up={mism(a_up, ck_up)} down={mism(a_down, ck_down)}/{a_down.numel()}; "
                 f"gate[0,:4] arena={a_gate[0, :4].tolist()} ckpt={ck_gate[0, :4].tolist()}")
        # And the GPU-resident slots: does slot s hold logical p2l[s]?
        layer = self._selftest_layer
        for sl in (0, 1, m.num_gpu_experts - 1):
            le = int(p2l[sl]) if p2l is not None else sl
            g_gate = layer.w13_weight[sl, :N].cpu()
            g_up = layer.w13_weight[sl, N:].cpu()
            g_down = layer.w2_weight[sl].cpu()
            _log(f"selftest GPU slot {sl} vs logical {le}: mismatches gate={int((g_gate != gu[le, r * N:(r + 1) * N]).sum())} "
                 f"up={int((g_up != gu[le, I + r * N:I + (r + 1) * N]).sum())} down={int((g_down != dn[le, :, r * N:(r + 1) * N]).sum())}"
                 + (f"; vs logical {sl}: gate={int((g_gate != gu[sl, r * N:(r + 1) * N]).sum())}" if le != sl else ""))

    @staticmethod
    def _physical_to_logical(layer_idx: int):
        from sglang.srt.eplb.expert_location_dispatch import (
            get_global_expert_location_metadata,
        )

        meta = get_global_expert_location_metadata()
        if meta is None:
            return None
        m = meta.physical_to_logical_map_cpu[layer_idx].to(torch.int64)
        return None if torch.equal(m, torch.arange(m.numel())) else m

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
        if slot.raw is None:
            wstride = arenas[0][1]
            assert all(a[1] == wstride for a in arenas)
            slot.raw = torch.empty((3, self.G, wstride), dtype=torch.uint8, device=self.device)
        wstride = slot.raw.shape[2]
        t0 = time.perf_counter()
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            for pi in range(3):
                slot.raw[pi, :n].view(-1).copy_(arenas[pi][0][base * wstride:(base + n) * wstride], non_blocking=True)
            slot.dma_done.record(self.transfer_stream)
        stats["dma_enq"] += time.perf_counter() - t0

    # -- one group ---------------------------------------------------------------

    def _group_body(self, layer, gi: int, base: int, n: int, g):
        """Copy the group into the [gate; up] / down slot tensors, run the
        bf16 MoE, accumulate; captured into a graph per (token count, group)."""
        slot = self.slots[gi % 2]
        N, K = self.N, self.K
        w_bytes = N * K * 2
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        raw = slot.raw[:, :n]
        # FlashInfer's CUTLASS MoE takes fc1 as [up; gate] rows (w31), the
        # order SGLang's swap_w13_to_w31 produces for it; [gate; up] gave
        # outputs with an error larger than the signal (09-14 refcheck).
        slot.w13_weight[:n, :N].copy_(raw[1, :, :w_bytes].view(torch.bfloat16).view(n, N, K))
        slot.w13_weight[:n, N:].copy_(raw[0, :, :w_bytes].view(torch.bfloat16).view(n, N, K))
        slot.w2_weight[:n].copy_(raw[2, :, :w_bytes].view(torch.bfloat16).view(n, K, N))
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        # Rows whose experts are all outside this group are not written by
        # the kernel; they must read as zero for the accumulate below.
        out.zero_()
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=slot.w13_weight[:n],
            fc2_expert_weights=slot.w2_weight[:n],
            output_dtype=torch.bfloat16,
            quant_scales=None,
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

    def _graph_inputs(self, T: int, x, physical_ids, topk_weights):
        """Persistent graph inputs for this token count."""
        g = self._gin.get(T)
        if g is None:
            g = {
                "x": x.clone(),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
            }
            self._gin[T] = g
        else:
            g["x"].copy_(x)
        g["ids"].copy_(physical_ids)
        g["w"].copy_(topk_weights)
        g["acc"].zero_()
        return g

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
        T = x.shape[0]
        layer_idx = method.kt_config.layer_idx
        if os.environ.get("KT_GPU_STREAM_SELFTEST") == "2" and layer_idx == min(self._arenas):
            # Per-call integrity probe: checksums of a CPU arena slot, a GPU
            # expert slot, the input and the ids.
            ar = self._arenas[layer_idx]
            wst = ar[0][1]
            a48 = ar[0][0][method.num_gpu_experts * wst:(method.num_gpu_experts + 1) * wst].view(torch.int32).sum().item()
            g0 = layer.w13_weight[0].view(torch.int32).sum().item()
            _log(f"rank{self.rank} probe layer{layer_idx} T={T} arena_slot{method.num_gpu_experts}={a48} gpu_slot0={g0} "
                 f"x={x.float().abs().sum().item():.3f} ids[:8]={physical_ids.reshape(-1)[:8].tolist()} "
                 f"idmax={int(physical_ids.max())} w={dispatch_output.topk_output.topk_weights.float().sum().item():.2f}")
        # One value for every chunk size: the bf16 kernel's own buffers follow
        # it, and per-T values produced garbage for T != 2048 (09-14).
        self._tune_tokens = max(next_power_of_2(T), int(os.environ.get("KT_GPU_STREAM_TUNE_TOKENS", "2048")))
        if self._workspace is None:
            # Sized once for the prefill chunk and never replaced: the group
            # graphs bake its address in.
            from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
            from sglang.srt.server_args import get_global_server_args

            Tmax = max(T, get_global_server_args().chunked_prefill_size or 0)
            nbytes = cutlass_fused_moe_workspace_size(
                Tmax, self.K, self.N, self.G * layer.moe_ep_size, physical_ids.shape[1],
                x_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                activation_type=self._activation_type, tp_size=layer.moe_tp_size, tp_rank=layer.moe_tp_rank,
                ep_size=layer.moe_ep_size, ep_rank=layer.moe_ep_rank,
                use_fused_finalize=self._use_fused_finalize, device=x.device)
            self._workspace = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {Tmax} tokens x {self.G} experts")
        topk_weights = dispatch_output.topk_output.topk_weights
        first = method.num_gpu_experts
        last = self._num_experts(layer_idx)
        groups = [list(range(b, min(b + self.G, last))) for b in range(first, last, self.G)]
        stats = {"dma_enq": 0.0, "graph_enq": 0.0}
        t0 = time.perf_counter()
        g = self._graph_inputs(T, x, physical_ids, topk_weights)
        cur = torch.cuda.current_stream(self.device)
        self._stage(layer_idx, 0, groups[0], stats)
        for gi, experts in enumerate(groups):
            slot = self.slots[gi % 2]
            t1 = time.perf_counter()
            if USE_GRAPH:
                graph = self._graph_for(layer, T, gi, experts[0], len(experts), g)
                cur.wait_event(slot.dma_done)
                graph.replay()
            else:
                cur.wait_event(slot.dma_done)
                self._group_body(layer, gi, experts[0], len(experts), g)
            slot.compute_done.record(cur)
            slot.compute_used = True
            stats["graph_enq"] += time.perf_counter() - t1
            if gi + 1 < len(groups):
                self._stage(layer_idx, gi + 1, groups[gi + 1], stats)
        self.calls += 1
        if TIMING:
            cur.synchronize()
            _log(f"rank{self.rank} layer{layer_idx} tokens={T} groups={len(groups)} "
                 f"dma_enq={stats['dma_enq']*1e3:.0f}ms graph_enq={stats['graph_enq']*1e3:.0f}ms "
                 f"total={(time.perf_counter()-t0)*1e3:.0f}ms")
        if os.environ.get("KT_GPU_STREAM_SELFTEST") == "2" and layer_idx == min(self._arenas):
            self._reference_check(method, layer_idx, x, physical_ids, topk_weights, g["acc"])
        # The caller adds this before the next layer zeroes it (same stream).
        return g["acc"]

    def _reference_check(self, method, layer_idx, x, ids, w, acc):
        """This rank's partial MoE output for the streamed experts, computed
        from the arenas with plain torch matmuls, against acc."""
        torch.cuda.current_stream(self.device).synchronize()
        N, K = self.N, self.K
        ar = self._arenas[layer_idx]
        wst = ar[0][1]
        first = method.num_gpu_experts
        ref = torch.zeros_like(acc, dtype=torch.float32)
        ids64 = ids.to(torch.int64)
        xf = x.float()
        used = 0
        for e in range(first, self._num_experts(layer_idx)):
            rows, cols = torch.where(ids64 == e)
            if rows.numel() == 0:
                continue
            used += 1
            gate = ar[0][0][e * wst:e * wst + N * K * 2].view(torch.bfloat16).view(N, K).to(self.device).float()
            up = ar[1][0][e * wst:e * wst + N * K * 2].view(torch.bfloat16).view(N, K).to(self.device).float()
            down = ar[2][0][e * wst:e * wst + K * N * 2].view(torch.bfloat16).view(K, N).to(self.device).float()
            h = xf[rows]
            a = torch.nn.functional.silu(h @ gate.t()) * (h @ up.t())
            o = (a @ down.t()) * w[rows, cols].float().unsqueeze(1)
            ref.index_add_(0, rows, o)
        d = (acc.float() - ref).abs()
        _log(f"rank{self.rank} refcheck layer{layer_idx} T={x.shape[0]} experts_used={used}: max|acc-ref|={d.max().item():.4f} "
             f"mean={d.mean().item():.5f} ref_mean|.|={ref.abs().mean().item():.5f} acc_mean|.|={acc.float().abs().mean().item():.5f} "
             f"rows_bad={int((d.max(1).values > 0.05 * (ref.abs().max(1).values + 1e-3)).sum())}/{x.shape[0]}")


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
