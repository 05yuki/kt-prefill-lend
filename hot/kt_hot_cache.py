"""Decode-adaptive hot experts for the KT hybrid with GGUF experts
(KT_HOT_DYNAMIC=1): the GPU slots of every layer follow the decode's own
routing instead of a static map.

Measured on a 38K brief + 512 generated tokens: a corpus map catches 0.29 /
0.42 / 0.53 of the decode routes with 32 / 64 / 96 slots, the decode's own
top-n 0.50 / 0.69 / 0.80 (tools/routing-prefill-vs-decode.py). Every route a
slot catches is one expert the CPU does not compute that layer.

How it works
- tables on the GPU: log2slot [L, E] (slot or -1), log2cpu [L, E] (-1 or the
  logical id itself), both int32 like the router's ids, and counts [L, E]; the
  captured decode graph only does table lookups and an index_add_ on them,
  so the policy runs outside the graph, between steps (model_runner.forward
  -> before_forward), and changes the tables in place.
- the CPU pools are in logical order (launch with KT_EXPERT_PLACEMENT=none);
  the CPU gets the logical ids with the resident ones set to -1, the GPU
  method the slot ids with the non-resident ones -1.
- a swap copies the expert's TP slice from the GGUF streamer's pinned banks
  (every expert of every layer, this rank's rows) into the slot; the lend's
  host mirror of the slot gets the same bytes, so the restore after a prefill
  keeps the swapped experts.
- policy every KT_HOT_INTERVAL decode (or MTP verify) steps: counts *= KT_HOT_DECAY; per
  layer, the coldest residents give way to the hottest non-residents when
  candidate > KT_HOT_MARGIN * resident, at most KT_HOT_SWAPS per layer.
- KT_HOT_SEED=<placement json> fills the slots at start from a map
  (physical_to_logical_map, first N per layer); without it they start as
  logical 0..N-1.
- KT_HOT_KV_LEND=1 (kt_hot_kv_lend.py): more slots per layer, ids N.., in free
  KV pages. A slot entry is the resident logical id, -1 for an empty slot the
  policy fills first, -2 for a slot that does not exist (not lent now). The
  kernels read every slot through per-layer pointer tables; base tables
  (log2slot_base / log2cpu_base) leave the lent residents on the CPU for the
  kernel path that cannot read the tables (kt_gguf_moe_multi.reads_table).
"""
import json
import os
import time
from typing import Dict, List, Optional

import torch

ENABLED = os.environ.get("KT_HOT_DYNAMIC") == "1"
INTERVAL = int(os.environ.get("KT_HOT_INTERVAL", "32"))
MAX_SWAPS = int(os.environ.get("KT_HOT_SWAPS", "8"))
DECAY = float(os.environ.get("KT_HOT_DECAY", "0.9"))
MARGIN = float(os.environ.get("KT_HOT_MARGIN", "1.5"))
SEED = os.environ.get("KT_HOT_SEED")
LOG_EVERY = int(os.environ.get("KT_HOT_LOG", "0"))  # rounds between log lines (0: only the first and every 64th)
KV_LEND = os.environ.get("KT_HOT_KV_LEND", "0") == "1"
KV_LEND_MAX = int(os.environ.get("KT_HOT_KV_LEND_MAX", "64"))  # lent slots per layer at most (table size)
# KT_HOT_FUSED_ROUTE=1: route() as one Triton kernel (the two table lookups and the count update) instead of
# index_add_ + two gathers (+ casts): 4-5 launches, ~10 us a layer on the critical path between the router and the
# CPU submit; integer tables, the counts stay exact integers, so the result is identical. =0 keeps the torch ops.
FUSED_ROUTE = os.environ.get("KT_HOT_FUSED_ROUTE", "1") == "1"
# KT_HOT_BANKS=own: build the cache's own pinned expert banks even when the GGUF streamer is loaded (otherwise
# the streamer's banks are shared; without the streamer the own banks are the only source)
OWN_BANKS = os.environ.get("KT_HOT_BANKS", "auto")
_ROUTE_BLOCK = 64  # routes per call at most for the fused kernel (decode / verify windows: tokens x top_k)


def _route_kernel_fn():
    import triton
    import triton.language as tl

    @triton.jit
    def _route_kernel(ids_ptr, slot_tab_ptr, cpu_tab_ptr, counts_ptr, slot_out_ptr, cpu_out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ids = tl.load(ids_ptr + offs, mask=m, other=0)
        tl.store(slot_out_ptr + offs, tl.load(slot_tab_ptr + ids, mask=m, other=-1), mask=m)
        tl.store(cpu_out_ptr + offs, tl.load(cpu_tab_ptr + ids, mask=m, other=-1), mask=m)
        tl.atomic_add(counts_ptr + ids, 1.0, mask=m)

    return _route_kernel


_route_kernel = None
EMPTY, ABSENT = -1, -2


def _log(msg):
    print(f"[kt-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HotCache:
    def __init__(self, num_layers: int, num_experts: int, slots: int, device: torch.device, rank: int):
        self.L, self.E, self.N, self.device, self.rank = num_layers, num_experts, slots, device, rank
        # int32 like the router's topk ids: route() is two lookups, no casts
        self.log2slot_cpu = torch.full((num_layers, num_experts), -1, dtype=torch.int32)
        self.log2slot = torch.full((num_layers, num_experts), -1, dtype=torch.int32, device=device)
        # the CPU's view: -1 for a resident expert, the logical id otherwise
        self.log2cpu_cpu = torch.arange(num_experts, dtype=torch.int32).repeat(num_layers, 1)
        self.log2cpu = self.log2cpu_cpu.to(device)
        self.counts = torch.zeros((num_layers, num_experts), dtype=torch.float32, device=device)
        self.ones = torch.ones((1,), dtype=torch.float32, device=device)
        self.S = slots + (KV_LEND_MAX if KV_LEND else 0)  # slot ids per layer (resident + lent)
        self.slot2log: List[List[int]] = [[EMPTY] * slots + [ABSENT] * (self.S - slots) for _ in range(num_layers)]
        # lent slot (layer, slot) -> (w13 view, w2 view) in borrowed KV pages
        self.lent_views: Dict[tuple, tuple] = {}
        if KV_LEND:
            self.log2slot_base_cpu = self.log2slot_cpu.clone()
            self.log2cpu_base_cpu = self.log2cpu_cpu.clone()
            self.log2slot_base = self.log2slot.clone()
            self.log2cpu_base = self.log2cpu.clone()
        self.layers: Dict[int, tuple] = {}  # layer_idx -> (hot method, FusedMoE layer)
        self._mirrors: Dict[tuple, Optional[torch.Tensor]] = {}
        self._counts_host = torch.zeros((num_layers, num_experts), dtype=torch.float32).pin_memory()
        self._counts_event: Optional[torch.cuda.Event] = None
        self._q8_banks: Dict[int, torch.Tensor] = {}
        self._own_banks: Dict[int, list] = {}
        self.ready = False
        self.step = 0
        self.rounds = 0
        self.swaps = 0
        self.bytes = 0
        self.swap_s = 0.0

    # -- graph side ----------------------------------------------------------

    def route(self, layer_idx: int, ids: torch.Tensor):
        """ids: logical [T, k] (>= 0). Returns (slot ids with -1 for CPU experts,
        logical ids with -1 for the resident ones). Captured in the graph."""
        flat = ids.reshape(-1)
        slot_tab, cpu_tab = self.log2slot, self.log2cpu
        if KV_LEND:
            from sglang.srt.layers.moe import kt_gguf_moe_multi as _multi

            if not _multi.reads_table(ids.shape[0]):
                # a kernel that addresses base + slot * stride: lent residents
                # are computed by the CPU in this call
                slot_tab, cpu_tab = self.log2slot_base, self.log2cpu_base
        if FUSED_ROUTE and flat.numel() <= _ROUTE_BLOCK and flat.is_contiguous():
            global _route_kernel
            if _route_kernel is None:
                _route_kernel = _route_kernel_fn()
            slot_ids = torch.empty_like(flat, dtype=torch.int32)
            cpu_ids = torch.empty_like(flat, dtype=torch.int32)
            _route_kernel[(1,)](flat, slot_tab[layer_idx], cpu_tab[layer_idx], self.counts[layer_idx], slot_ids, cpu_ids,
                                flat.numel(), BLOCK=_ROUTE_BLOCK, num_warps=1)
            return slot_ids.view_as(ids), cpu_ids.view_as(ids)
        self.counts[layer_idx].index_add_(0, flat, self.ones.expand(flat.numel()))
        return slot_tab[layer_idx][ids], cpu_tab[layer_idx][ids]

    def push_tables(self) -> None:
        """The host tables to the GPU on the current stream."""
        self.log2slot.copy_(self.log2slot_cpu, non_blocking=True)
        self.log2cpu.copy_(self.log2cpu_cpu, non_blocking=True)
        if KV_LEND:
            self.log2slot_base.copy_(self.log2slot_base_cpu, non_blocking=True)
            self.log2cpu_base.copy_(self.log2cpu_base_cpu, non_blocking=True)

    # -- filling -------------------------------------------------------------

    def _bank(self, layer_idx: int):
        """This rank's slice of every expert of the layer (gate, up, down), pinned, physical order: the GGUF
        streamer's banks when it is loaded (one pinned copy serves both), otherwise the cache's own, built from the
        hot method's GGUF reader (KT_HOT_BANKS=own forces these; the cache does not need the streamer)."""
        if OWN_BANKS != "own":
            try:
                from sglang.srt.layers.moe import kt_stream_prefill_gguf as G
            except ImportError:
                G = None
            inst = getattr(G, "_instance", None)
            if inst is not None:
                return inst._bank(layer_idx)
        b = self._own_banks.get(layer_idx)
        if b is not None:
            return b
        t0 = time.time()
        method, _layer = self.layers[layer_idx]
        rd, E, N, K, r, world = method.reader, method.reader.E, method.N, method.K, method.rank, method.world
        from sglang.srt.layers.moe.kt_gguf_hot import BLOCK_INFO

        p2l = None
        try:
            from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata

            meta = get_global_expert_location_metadata()
            if meta is not None:
                m = meta.physical_to_logical_map_cpu[layer_idx].to(torch.int64)
                if not torch.equal(m, torch.arange(m.numel(), dtype=torch.int64)):
                    p2l = m
        except ImportError:
            pass
        banks = []
        for proj in ("gate", "up", "down"):
            data, qt = rd._raw(f"blk.{layer_idx}.ffn_{proj}_exps.weight")
            blck, ts = BLOCK_INFO[qt]
            if proj != "down":
                v = data.view(E, N * world, (K // blck) * ts)[:, N * r:N * (r + 1), :]
            else:
                v = data.view(E, K, (N * world) // blck, ts)[:, :, (N * r) // blck:(N * (r + 1)) // blck, :]
            v = v.reshape(E, -1)
            if p2l is not None:
                v = v[p2l]
            h = torch.empty((E, v.shape[1]), dtype=torch.uint8, pin_memory=True)
            h.copy_(v)
            banks.append(h)
            del data
        self._own_banks[layer_idx] = banks
        if layer_idx == min(self.layers):
            _log(f"rank {self.rank}: layer {layer_idx} own bank {sum(b.numel() for b in banks) / 1e9:.2f} GB pinned in "
                 f"{time.time() - t0:.1f} s (no streamer{', placement map' if p2l is not None else ''})")
        return banks

    def _mirror(self, layer_idx: int, param: torch.Tensor) -> Optional[torch.Tensor]:
        """The lend's pinned host copy of this parameter (same shape), or None.
        Resolved once per parameter after the lend has mirrored."""
        from sglang.srt.layers.moe import kt_lend

        key = (layer_idx, param.data_ptr())
        if key in self._mirrors:
            return self._mirrors[key]
        st = kt_lend.STATE
        if "host" not in st or "dev_spans" not in st:
            return None  # not mirrored yet (seeding at load) or lend off
        found = None
        for i, dspan in enumerate(st["dev_spans"]):
            lo = dspan.data_ptr()
            if lo <= param.data_ptr() < lo + dspan.numel():
                off = param.data_ptr() - lo + st["bases"][i]
                found = st["host"][off:off + param.numel() * param.element_size()].view(param.dtype).view(param.shape)
                break
        self._mirrors[key] = found
        return found

    def _down_source(self, layer_idx: int, method, down_bank: torch.Tensor) -> torch.Tensor:
        """The bank to copy a slot's down matrix from: the raw bank, or for the
        Q2_0 layers a Q8_0 repack of the whole layer built on first use
        (every expert, 0.45 GB pinned per such layer)."""
        if method.types[2] == method.type2:
            return down_bank
        b = self._q8_banks.get(layer_idx)
        if b is None:
            from sglang.srt.layers.moe.kt_gguf_hot import q2_0_to_q8_0

            t0 = time.time()
            b = q2_0_to_q8_0(down_bank).pin_memory()
            self._q8_banks[layer_idx] = b
            _log(f"rank {self.rank}: layer {layer_idx} down bank repacked Q2_0 -> Q8_0 for the slots "
                 f"({b.numel() / 1e9:.2f} GB pinned, {time.time() - t0:.1f} s)")
        return b

    def fill(self, layer_idx: int, slot: int, logical: int) -> None:
        method, layer = self.layers[layer_idx]
        gate, up, down = self._bank(layer_idx)
        N = method.N
        w13, w2 = layer.w13_qweight.data, layer.w2_qweight.data
        g = gate[logical].view(N, -1)
        u = up[logical].view(N, -1)
        d = self._down_source(layer_idx, method, down)[logical].view(w2.shape[1], -1)
        if slot >= self.N:
            # a lent slot: KV pages, no lend mirror (lent slots are given back
            # before every streamed prefill, kt_hot_kv_lend.py)
            v13, v2 = self.lent_views[(layer_idx, slot)]
            v13[:N].copy_(g, non_blocking=True)
            v13[N:].copy_(u, non_blocking=True)
            v2.copy_(d, non_blocking=True)
            self.bytes += g.numel() + u.numel() + d.numel()
            return
        w13[slot, :N].copy_(g, non_blocking=True)
        w13[slot, N:].copy_(u, non_blocking=True)
        w2[slot].copy_(d, non_blocking=True)
        m13, m2 = self._mirror(layer_idx, layer.w13_qweight), self._mirror(layer_idx, layer.w2_qweight)
        if m13 is not None:
            m13[slot, :N].copy_(g)
            m13[slot, N:].copy_(u)
        if m2 is not None:
            m2[slot].copy_(d)
        self.bytes += g.numel() + u.numel() + d.numel()

    def _set(self, layer_idx: int, slot: int, logical: int) -> None:
        """Slot <- logical (>= 0), or EMPTY / ABSENT. Residency is carried by
        the ids alone: an expert is in log2slot or in log2cpu, never both,
        never neither (check_tables)."""
        old = self.slot2log[layer_idx][slot]
        if old >= 0:
            self.log2slot_cpu[layer_idx, old] = -1
            self.log2cpu_cpu[layer_idx, old] = old
            if KV_LEND and slot < self.N:
                self.log2slot_base_cpu[layer_idx, old] = -1
                self.log2cpu_base_cpu[layer_idx, old] = old
        self.slot2log[layer_idx][slot] = logical
        if logical < 0:
            return
        self.log2slot_cpu[layer_idx, logical] = slot
        self.log2cpu_cpu[layer_idx, logical] = -1
        if KV_LEND and slot < self.N:
            self.log2slot_base_cpu[layer_idx, logical] = slot
            self.log2cpu_base_cpu[layer_idx, logical] = -1

    def check_tables(self) -> str:
        """Every expert of every layer exactly once: on a slot that holds it
        (full and base tables, host and device) or on the CPU. Returns '' or
        the first violation."""
        pairs = [("full", self.log2slot_cpu, self.log2cpu_cpu, self.log2slot, self.log2cpu, False)]
        if KV_LEND:
            pairs.append(("base", self.log2slot_base_cpu, self.log2cpu_base_cpu, self.log2slot_base,
                          self.log2cpu_base, True))
        for name, s_cpu, c_cpu, s_dev, c_dev, base in pairs:
            if not torch.equal(s_dev.cpu(), s_cpu) or not torch.equal(c_dev.cpu(), c_cpu):
                return f"{name}: device tables differ from the host's"
            if not bool(((s_cpu >= 0) ^ (c_cpu >= 0)).all()):
                return f"{name}: an expert both or neither on a slot and on the CPU"
            if not bool(((c_cpu < 0) | (c_cpu == torch.arange(self.E, dtype=torch.int32))).all()):
                return f"{name}: CPU id is not the logical id"
            for li in range(self.L):
                for e in (s_cpu[li] >= 0).nonzero().flatten().tolist():
                    sl = int(s_cpu[li, e])
                    if self.slot2log[li][sl] != e or (base and sl >= self.N):
                        return f"{name}: layer {li} expert {e} -> slot {sl} holding {self.slot2log[li][sl]}"
                for sl, e in enumerate(self.slot2log[li]):
                    if e >= 0 and (int(self.log2slot_cpu[li, e]) != sl):
                        return f"{name}: layer {li} slot {sl} holds {e} but the table says {int(self.log2slot_cpu[li, e])}"
        return ""

    # -- lent slots (kt_hot_kv_lend.py) -------------------------------------------

    def lend_slot(self, layer_idx: int, slot: int, v13: torch.Tensor, v2: torch.Tensor) -> None:
        assert self.slot2log[layer_idx][slot] == ABSENT, (layer_idx, slot)
        self.lent_views[(layer_idx, slot)] = (v13, v2)
        self.slot2log[layer_idx][slot] = EMPTY

    def take_back_slot(self, layer_idx: int, slot: int) -> int:
        """The slot's expert goes back to the CPU (host tables; push_tables
        before the next forward). Returns the expert it held or -1."""
        old = self.slot2log[layer_idx][slot]
        assert old != ABSENT, (layer_idx, slot)
        self._set(layer_idx, slot, ABSENT)
        del self.lent_views[(layer_idx, slot)]
        return old

    def fill_empty(self, counts: torch.Tensor, seed_rank: Optional[torch.Tensor]) -> int:
        """Fill every EMPTY slot with the hottest non-residents of its layer
        (counts [L, E] on the host; ties by the seed map's order). Host tables
        only; the caller pushes them."""
        n = 0
        for li in sorted(self.layers):
            empty = [s_ for s_, e in enumerate(self.slot2log[li]) if e == EMPTY]
            if not empty:
                continue
            score = counts[li].double()
            if seed_rank is not None:
                score = score + (self.E - seed_rank[li].double()) / (self.E + 1) * 1e-3
            res = [e for e in self.slot2log[li] if e >= 0]
            if res:
                score[torch.tensor(res)] = float("-inf")
            for s_, c in zip(empty, torch.topk(score, len(empty)).indices.tolist()):
                self._set(li, s_, c)
                self.fill(li, s_, c)
                n += 1
        return n

    def seed(self, placement: Optional[str]) -> None:
        """Initial residents: the map's first N per layer (filled from the
        banks), or logical 0..N-1 (what the loader put there)."""
        t0 = time.time()
        p2l = json.load(open(placement))["physical_to_logical_map"] if placement else None
        for li in sorted(self.layers):
            for s in range(self.N):
                e = int(p2l[li][s]) if p2l else s
                self._set(li, s, e)
                if p2l and e != s:
                    self.fill(li, s, e)
        self.push_tables()
        torch.cuda.current_stream(self.device).synchronize()
        self.ready = True
        _log(f"rank {self.rank}: {len(self.layers)} layers x {self.N} slots seeded from "
             f"{placement or 'logical order'} in {time.time() - t0:.1f} s ({self.bytes / 1e9:.2f} GB copied); "
             f"interval {INTERVAL}, max {MAX_SWAPS} swaps/layer, decay {DECAY}, margin {MARGIN}")
        self.bytes = 0

    # -- policy, between steps --------------------------------------------------

    def before_forward(self, forward_batch) -> None:
        # decode steps, and under MTP the target's verify steps (the decode of a
        # speculative serve; is_decode() alone never ran the policy there)
        mode = forward_batch.forward_mode
        if not self.ready or not (mode.is_decode() or mode.is_target_verify()):
            return
        self.step += 1
        if self.step % INTERVAL:
            return
        t0 = time.time()
        # the counts come through a pinned buffer one round late, so the policy
        # never waits for the forward stream (a .cpu() here would block on the
        # step in flight)
        if self._counts_event is not None and not self._counts_event.query():
            return
        counts = self._counts_host.clone() if self._counts_event is not None else None
        self._counts_host.copy_(self.counts, non_blocking=True)
        self.counts.mul_(DECAY)
        if self._counts_event is None:
            self._counts_event = torch.cuda.Event()
        self._counts_event.record(torch.cuda.current_stream(self.device))
        if counts is None:
            return
        # all layers at once: coldest residents vs hottest non-residents. An
        # EMPTY slot scores -1 (filled by any candidate seen at all), an ABSENT
        # one +inf (never chosen); without lent slots res has neither
        res = torch.tensor(self.slot2log)  # [L, S]
        score_res = counts.gather(1, res.clamp(min=0))
        score_res = torch.where(res == EMPTY, torch.full_like(score_res, -1.0), score_res)
        score_res = torch.where(res == ABSENT, torch.full_like(score_res, float("inf")), score_res)
        mask = torch.ones((counts.shape[0], counts.shape[1] + 1), dtype=torch.bool)
        mask.scatter_(1, torch.where(res >= 0, res, torch.full_like(res, counts.shape[1])), False)
        mask = mask[:, :-1]
        cand_score, cand_idx = torch.topk(torch.where(mask, counts, torch.full_like(counts, -1.0)), MAX_SWAPS, dim=1)
        order = torch.argsort(score_res, dim=1)[:, :MAX_SWAPS]  # coldest slots first
        rs = score_res.gather(1, order)
        do = ((cand_score > 0) & (cand_score > MARGIN * rs)).to(torch.int64).cumprod(1).bool()
        swaps = 0
        for li, j in do.nonzero().tolist():
            if li not in self.layers:
                continue
            s_, c = int(order[li, j]), int(cand_idx[li, j])
            self._set(li, s_, c)
            self.fill(li, s_, c)
            swaps += 1
        if swaps:
            self.push_tables()
        self.rounds += 1
        self.swaps += swaps
        self.swap_s += time.time() - t0
        if self.rounds == 1 or (LOG_EVERY and self.rounds % LOG_EVERY == 0) or self.rounds % 64 == 0:
            resident = sum(float(counts[li][torch.tensor([e for e in self.slot2log[li] if e >= 0], dtype=torch.long)].sum())
                           for li in self.layers)
            total = float(counts.sum())
            _log(f"rank {self.rank}: round {self.rounds} (step {self.step}): {swaps} swaps this round, "
                 f"{self.swaps} total, {self.bytes / 1e9:.2f} GB, {self.swap_s * 1e3 / self.rounds:.1f} ms/round; "
                 f"GPU share of recent routes {resident / max(total, 1):.3f}")


_cache: Optional[HotCache] = None


def register(method, layer) -> Optional[HotCache]:
    """Called by the KT wrapper on its first decode apply of each layer; the
    cache is built on the first call and seeded once every layer is known."""
    global _cache
    if not ENABLED:
        return None
    kt = method.kt_config
    if _cache is None:
        if kt.num_gpu_experts <= 0:
            raise RuntimeError("KT_HOT_DYNAMIC=1 needs --kt-num-gpu-experts > 0")
        _cache = HotCache(kt.num_layers, method.global_num_experts, kt.num_gpu_experts,
                          layer.w13_qweight.device, method.tp_rank)
    c = _cache
    if kt.layer_idx not in c.layers:
        c.layers[kt.layer_idx] = (method.gpu_method, layer)
        if len(c.layers) == kt.num_layers:
            c.seed(SEED)
    return c if c.ready else None


def before_forward(forward_batch) -> None:
    if _cache is not None:
        _cache.before_forward(forward_batch)
