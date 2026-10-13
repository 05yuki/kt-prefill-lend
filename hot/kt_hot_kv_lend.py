"""Hot experts in the KV pool's free pages (KT_HOT_KV_LEND=1, with the dynamic
hot cache KT_HOT_DYNAMIC=1 and the GGUF hot method).

The KV always wins:
- only pages in the allocator's free list are lent, never pages the radix tree
  holds, and the pool's size never changes: the lent pages count as available
  (available_size), so admission, eviction, retraction and the idle leak check
  see the pool as if nothing were lent;
- the lent region is the top of the page range, whole groups of P pages, page
  ids above a boundary B. Lending moves B down over groups whose every page is
  free (and keeps KT_HOT_KV_LEND_RESERVE tokens free below it); any alloc /
  alloc_extend / alloc_decode that needs more pages than the free list holds
  takes lent groups back first (B moves up): a device-wide sync (every forward
  that may read those slots has finished), the slots' experts back to the CPU
  in the tables, the group's pages zeroed in every K/V tensor (expert bytes
  read as bf16 KV could be Inf/NaN, and page-wise attention kernels load before
  they mask), then the pages join the free list, all before the kernel that
  computes the new locations is launched;
- a streamed prefill (the lend's switch to prefill) gives every lent group
  back; clear() (flush_cache) too.
Lending happens outside any graph, at the start of a non-streamed extend of the
target model and at the first decode / verify forward after an extend, a
streamed prefill or a clear: one device sync each, never per decode step.

A group's pages are, in each K and V tensor of every pool on this allocator (the
target's full-attention layers and the MTP draft's layer), one contiguous byte
range; it holds floor(P * page_bytes / slot_bytes) expert slots (w13 then w2,
256-byte aligned; slot_bytes from the largest layer). New slots go to the layers
with the fewest lent slots, as ids N.. (N = KT_GPU_EXPERTS), and are filled at
once with the hottest non-residents of the layer (decayed route counts; the seed
map's order breaks ties). The hot method reads every slot through per-layer
int64 pointer tables (layer.kt_kv_ptr13 / kt_kv_ptr2, fixed addresses, so the
captured graphs stay valid); a kernel that cannot read them gets ids from the
cache's base tables, where lent residents are CPU experts. Residency is carried
by the ids alone (the CPU MoE has no prefix mask under the dynamic cache).

KT_HOT_KV_LEND_CHECK=1: after every lend / take-back the tables are checked
(every expert exactly once, device == host), and before a group is zeroed each
of its slots is compared with the bank it was filled from (no KV write ever
reached a lent page). Also, a streamed prefill takes everything back already
in its alloc_extend and puts those pages first in the free list, so its KV
lands in pages that held experts (a one-chunk streamed prefill repeats bit for
bit across launches: the paired NLL's streamed set then tests that no stale
expert byte is read as KV).
"""
import os
import time
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.moe import kt_hot_cache

ENABLED = kt_hot_cache.KV_LEND
RESERVE = int(os.environ.get("KT_HOT_KV_LEND_RESERVE", "2048"))  # tokens kept free below the lent region
CHECK = os.environ.get("KT_HOT_KV_LEND_CHECK", "0") == "1"
RELEASE_AT_PREFILL = os.environ.get("KT_HOT_KV_LEND_RELEASE_AT_PREFILL", "1") == "1"
# 0: lend and take back pages but leave the slots empty (with the policy frozen
# the routing is then exactly the feature-off one: a test of the page handling)
FILL = os.environ.get("KT_HOT_KV_LEND_FILL", "1") == "1"
# a slot goes to a layer only while its lent count is within SLACK of the
# smallest: without it the space a large slot leaves at a range's end goes to
# the layers with the smallest slots every time (TP1: 11-53 per layer)
SLACK = int(os.environ.get("KT_HOT_KV_LEND_SLACK", "2"))
ALIGN = 256

_runners: List = []  # (is_draft, pool, allocator)
_lender: Optional["Lender"] = None
_tables_ready = False
STATS = {"lends": 0, "take_backs": 0, "released_at_prefill": 0, "groups_lent": 0, "groups_taken": 0}


def _log(msg):
    print(f"[kt-kv-lend {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _kv_tensors(pool):
    kvp = getattr(pool, "full_kv_pool", None) or pool
    out = []
    for name in ("k_buffer", "v_buffer"):
        bufs = getattr(kvp, name, None)
        if isinstance(bufs, (list, tuple)):
            out += list(bufs)
    return kvp, out


# -- pointer tables ------------------------------------------------------------------

def _make_tables(cache) -> None:
    """layer.kt_kv_ptr13 / kt_kv_ptr2: one weight address per slot id (resident
    slots at base + slot * stride, lent ones set when lent). Built before the
    graphs are captured; resident entries are rewritten whenever checked."""
    global _tables_ready
    for li, (method, layer) in cache.layers.items():
        w13, w2 = layer.w13_qweight, layer.w2_qweight
        if getattr(layer, "kt_kv_ptr13", None) is None:
            layer.kt_kv_ptr13_host = torch.zeros(cache.S, dtype=torch.int64)
            layer.kt_kv_ptr2_host = torch.zeros(cache.S, dtype=torch.int64)
            layer.kt_kv_ptr13 = torch.zeros(cache.S, dtype=torch.int64, device=w13.device)
            layer.kt_kv_ptr2 = torch.zeros(cache.S, dtype=torch.int64, device=w13.device)
        h13, h2 = layer.kt_kv_ptr13_host, layer.kt_kv_ptr2_host
        s13, s2 = w13[0].numel() * w13.element_size(), w2[0].numel() * w2.element_size()
        n = w13.shape[0]
        h13[:n] = w13.data_ptr() + torch.arange(n, dtype=torch.int64) * s13
        h2[:n] = w2.data_ptr() + torch.arange(n, dtype=torch.int64) * s2
        layer.kt_kv_base = (w13.data_ptr(), w2.data_ptr())
        layer.kt_kv_ptr13.copy_(h13, non_blocking=True)
        layer.kt_kv_ptr2.copy_(h2, non_blocking=True)
    _tables_ready = True


def _check_bases(cache) -> None:
    for li, (method, layer) in cache.layers.items():
        if layer.kt_kv_base != (layer.w13_qweight.data_ptr(), layer.w2_qweight.data_ptr()):
            raise RuntimeError(f"KT_HOT_KV_LEND: layer {li}'s hot slot tensors moved after the pointer tables were built")


# -- the lender ----------------------------------------------------------------------

class Lender:
    def __init__(self, alloc, pools, cache):
        self.alloc, self.cache = alloc, cache
        self.ps = alloc.page_size
        self.n = alloc.num_pages
        self.tensors = []  # (flat uint8 view, page bytes)
        rows = self.n * self.ps + self.ps
        for pool in pools:
            kvp, ts = _kv_tensors(pool)
            for t in ts:
                if not (t.is_cuda and t.is_contiguous() and t.nbytes % (self.n + 1) == 0
                        and t.shape[0] in (rows, self.n + 1)):
                    _log(f"skipping a KV tensor {tuple(t.shape)} {t.dtype}: not {self.n + 1} contiguous pages")
                    continue
                self.tensors.append((t.view(-1).view(torch.uint8), t.nbytes // (self.n + 1)))
        if not self.tensors:
            raise RuntimeError("KT_HOT_KV_LEND: no KV tensors to lend")
        # slot layout: w13 then w2, each layer its own size (the types differ:
        # 1.5-2.7 MB at TP1); a group's byte range in each tensor is packed
        # with the slots of the layers that have the fewest lent slots. The
        # group size P is the one that gives the most slots when every group
        # is lent (simulated with the same packing)
        self.lsb = {li: _align(layer.w13_qweight[0].numel()) + _align(layer.w2_qweight[0].numel())
                    for li, (method, layer) in cache.layers.items()}
        self.sb = max(self.lsb.values())
        pbmax = max(pb for _, pb in self.tensors)
        pmin = -(-self.sb // pbmax)
        best = None
        for P in range(pmin, 4 * pmin + 1):
            self.P = P
            cnt = {li: 0 for li in cache.layers}
            first = len(self._pack(1, cnt))
            total = first + sum(len(self._pack(1, cnt)) for _ in range(self.n // P - 1))
            if best is None or total > best[1]:
                best = (P, total, first)
        self.P, total_slots, self.per_group = best
        self.gmax = self.n // self.P
        self.lent = 0  # groups lent, from the top
        self.group_slots: List[List[tuple]] = []  # per lent group (top first): (layer, slot)
        self.reserve_pages = -(-RESERVE // self.ps)
        self.seed_rank = self._seed_rank()
        _log(f"{len(pools)} pools on the allocator, {len(self.tensors)} K/V tensors, page {self.ps} tokens "
             f"({min(pb for _, pb in self.tensors)}-{pbmax} B a page per tensor), {self.n} pages; slot "
             f"{min(self.lsb.values())}-{self.sb} B, "
             f"group {self.P} pages = {self.per_group} slots (the first), at most {self.gmax} groups "
             f"({total_slots} slots, {total_slots / max(1, len(cache.layers)):.1f} per layer); "
             f"reserve {self.reserve_pages} pages, at most {kt_hot_cache.KV_LEND_MAX} lent slots per layer")

    def _seed_rank(self) -> Optional[torch.Tensor]:
        """Rank of each expert in the seed map per layer (0 = first), for ties."""
        path = kt_hot_cache.SEED
        if not path:
            return None
        import json

        p2l = json.load(open(path))["physical_to_logical_map"]
        c = self.cache
        r = torch.full((c.L, c.E), c.E, dtype=torch.int64)
        for li in range(min(c.L, len(p2l))):
            seen = {}
            for i, e in enumerate(p2l[li]):
                seen.setdefault(int(e), i)
            for e, i in seen.items():
                r[li, e] = i
        return r

    @property
    def boundary(self) -> int:
        return self.n - self.lent * self.P  # pages above it are lent

    def lent_pages(self) -> int:
        return self.lent * self.P

    def _group_pages(self, j: int):
        hi = self.n - j * self.P
        return hi - self.P + 1, hi  # inclusive

    def _pack(self, lo: int, cnt: Dict[int, int]):
        """Slots for the group starting at page lo: [(tensor, byte offset,
        layer)], each next slot for the layer with the fewest lent slots that
        still fits in the tensor's range (cnt is updated)."""
        out = []
        cap = kt_hot_cache.KV_LEND_MAX
        for ti, (flat, pb) in enumerate(self.tensors):
            off, end = lo * pb, (lo + self.P) * pb
            while True:
                pick = None
                low = min(cnt.values())
                for li in sorted(cnt, key=lambda l: (cnt[l], l)):
                    if cnt[li] > low + SLACK:
                        break
                    if cnt[li] < cap and off + self.lsb[li] <= end:
                        pick = li
                        break
                if pick is None:
                    break
                out.append((ti, off, pick))
                off += self.lsb[pick]
                cnt[pick] += 1
        return out

    def _free_count(self) -> int:
        a = self.alloc
        return len(a.free_pages) + (len(a.release_pages) if a.release_pages is not None else 0)

    # lend ---------------------------------------------------------------------------

    def try_lend(self, why: str) -> None:
        c = self.cache
        if self.lent >= self.gmax:
            return
        if all(sum(1 for e in c.slot2log[li][c.N:] if e != kt_hot_cache.ABSENT) >= kt_hot_cache.KV_LEND_MAX
               for li in c.layers):
            return
        t0 = time.perf_counter()
        a = self.alloc
        dev = a.free_pages.device
        torch.cuda.synchronize(dev)
        B = self.boundary
        m = B // self.P - 1  # whole groups below B, page 0 never lent
        free_n = self._free_count()
        room = (free_n - self.reserve_pages) // self.P
        k = 0
        if m > 0 and room > 0:
            mark = torch.zeros(self.n + 2, dtype=torch.bool, device=dev)
            mark[a.free_pages] = True
            if a.release_pages is not None and len(a.release_pages):
                mark[a.release_pages] = True
            x = mark[B - m * self.P + 1:B + 1].view(m, self.P).all(1).flip(0)  # x[0]: the group just below B
            full = x.cpu().tolist()
            while k < min(m, room) and full[k]:
                k += 1
        if k == 0:
            self._summary(why, 0, 0, time.perf_counter() - t0, quiet=True)
            return
        newB = B - k * self.P
        before = len(a.free_pages)
        a.free_pages = a.free_pages[a.free_pages <= newB]
        removed = before - len(a.free_pages)
        if a.release_pages is not None and len(a.release_pages):
            rb = len(a.release_pages)
            a.release_pages = a.release_pages[a.release_pages <= newB]
            removed += rb - len(a.release_pages)
        if removed != k * self.P:
            raise RuntimeError(f"KT_HOT_KV_LEND: {removed} free pages above {newB}, expected {k * self.P}")
        _check_bases(c)
        changed = set()
        new_slots = 0
        cnt = {li: sum(1 for e in c.slot2log[li][c.N:] if e != kt_hot_cache.ABSENT) for li in c.layers}
        for j in range(self.lent, self.lent + k):
            slots = []
            for ti, off, li in self._pack(self._group_pages(j)[0], cnt):
                slot = c.slot2log[li].index(kt_hot_cache.ABSENT, c.N)
                method, layer = c.layers[li]
                flat = self.tensors[ti][0]
                w13, w2 = layer.w13_qweight, layer.w2_qweight
                b13, b2 = w13[0].numel(), w2[0].numel()
                o2 = off + _align(b13)
                v13 = flat[off:off + b13].view(w13.shape[1:])
                v2 = flat[o2:o2 + b2].view(w2.shape[1:])
                c.lend_slot(li, slot, v13, v2)
                layer.kt_kv_ptr13_host[slot] = v13.data_ptr()
                layer.kt_kv_ptr2_host[slot] = v2.data_ptr()
                changed.add(li)
                slots.append((li, slot))
            self.group_slots.append(slots)
            new_slots += len(slots)
        self.lent += k
        for li in changed:
            layer = c.layers[li][1]
            layer.kt_kv_ptr13.copy_(layer.kt_kv_ptr13_host, non_blocking=True)
            layer.kt_kv_ptr2.copy_(layer.kt_kv_ptr2_host, non_blocking=True)
        filled = c.fill_empty(c.counts.float().cpu(), self.seed_rank) if FILL else 0
        c.push_tables()
        STATS["lends"] += 1
        STATS["groups_lent"] += k
        if CHECK:
            torch.cuda.synchronize(dev)
            err = c.check_tables()
            if err:
                raise RuntimeError(f"KT_HOT_KV_LEND check after lend: {err}")
        self._summary(why, k, filled, time.perf_counter() - t0)

    # take back ----------------------------------------------------------------------

    def take_back(self, groups: int, why: str, first: bool = False) -> None:
        groups = min(groups, self.lent)
        if groups <= 0:
            return
        t0 = time.perf_counter()
        c = self.cache
        a = self.alloc
        dev = a.free_pages.device
        # every forward that may read these slots has finished
        torch.cuda.synchronize(dev)
        if CHECK:
            self._check_slots(range(self.lent - groups, self.lent))
        oldB = self.boundary
        n_slots = 0
        for j in range(self.lent - 1, self.lent - 1 - groups, -1):
            for li, slot in self.group_slots[j]:
                c.take_back_slot(li, slot)
                n_slots += 1
        del self.group_slots[self.lent - groups:]
        self.lent -= groups
        newB = self.boundary
        c.push_tables()
        for flat, pb in self.tensors:
            flat[(oldB + 1) * pb:(newB + 1) * pb].zero_()
        torch.cuda.current_stream(dev).synchronize()
        back = torch.arange(oldB + 1, newB + 1, dtype=a.free_pages.dtype, device=dev)
        a.free_pages = torch.cat((back, a.free_pages) if first else (a.free_pages, back))
        STATS["take_backs"] += 1
        STATS["groups_taken"] += groups
        if CHECK:
            err = c.check_tables()
            if err:
                raise RuntimeError(f"KT_HOT_KV_LEND check after take-back: {err}")
        _log(f"take back {groups} groups ({n_slots} slots, pages {oldB + 1}-{newB}) for {why} in "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms; {self.lent} groups lent; take-backs {STATS['take_backs']}, "
             f"groups taken {STATS['groups_taken']} / lent {STATS['groups_lent']}")
        global _want_lend
        _want_lend = True

    def _check_slots(self, js) -> None:
        """Each lent slot still holds the bytes of the expert it was filled with."""
        c = self.cache
        bad = 0
        n = 0
        for j in js:
            for li, slot in self.group_slots[j]:
                e = c.slot2log[li][slot]
                if e < 0:
                    continue
                method, layer = c.layers[li]
                gate, up, down = c._bank(li)
                N = method.N
                v13, v2 = c.lent_views[(li, slot)]
                d = c._down_source(li, method, down)[e]
                ok = (torch.equal(v13[:N].cpu(), gate[e].view(N, -1)) and torch.equal(v13[N:].cpu(), up[e].view(N, -1))
                      and torch.equal(v2.cpu().view(-1), d.reshape(-1)))
                bad += not ok
                n += 1
        if bad:
            raise RuntimeError(f"KT_HOT_KV_LEND: {bad} of {n} lent slots changed while lent (a KV write reached a lent page)")
        _log(f"check: {n} lent slots intact before take-back")

    def _summary(self, why, k, filled, dt, quiet=False):
        c = self.cache
        per = [sum(1 for e in c.slot2log[li][c.N:] if e != kt_hot_cache.ABSENT) for li in sorted(c.layers)]
        line = (f"lend at {why}: +{k} groups -> {self.lent} lent ({self.lent * self.P} pages, "
                f"{sum(per)} slots, per layer {min(per)}-{max(per)}, {filled} filled) in {dt * 1e3:.1f} ms; "
                f"free {self._free_count()} of {self.n} pages")
        if quiet:
            STATS["quiet"] = STATS.get("quiet", 0) + 1
            if STATS["quiet"] in (1, 10, 100) or STATS["quiet"] % 1000 == 0:
                _log(line + f" (no group free, {STATS['quiet']} such)")
            return
        _log(line)


# -- allocator wrappers --------------------------------------------------------------

def _wrap_allocator(a) -> None:
    from sglang.srt.utils import get_num_new_pages

    if getattr(a, "_kt_kv_lend", False):
        return
    a._kt_kv_lend = True
    orig = {k: getattr(a, k) for k in ("alloc", "alloc_extend", "alloc_decode", "available_size", "clear")}

    def ensure(pages: int, why: str) -> None:
        L = _lender
        if L is None or L.lent == 0:
            return
        free = L._free_count()
        if pages > free:
            L.take_back(-(-(pages - free) // L.P), why)

    def alloc(need_size: int):
        L = _lender
        if L is not None and L.lent:
            ensure(need_size // a.page_size, "alloc")
        return orig["alloc"](need_size)

    def alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens,
                     num_new_pages=None):
        L = _lender
        if L is not None and L.lent and CHECK and extend_num_tokens >= _threshold():
            L.take_back(L.lent, "streamed prefill (check: at its alloc, pages first)", first=True)
        if L is not None and L.lent:
            bound = extend_num_tokens // a.page_size + len(prefix_lens) + 1
            if bound > L._free_count():
                need = num_new_pages if num_new_pages is not None else get_num_new_pages(
                    seq_lens=seq_lens_cpu, page_size=a.page_size, prefix_lens=prefix_lens_cpu)
                ensure(int(need), "alloc_extend")
        return orig["alloc_extend"](prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc,
                                    extend_num_tokens, num_new_pages)

    def alloc_decode(seq_lens, seq_lens_cpu, last_loc):
        L = _lender
        if L is not None and L.lent and len(seq_lens) > L._free_count():
            need = get_num_new_pages(seq_lens=seq_lens_cpu, page_size=a.page_size, decode=True)
            ensure(int(need), "alloc_decode")
        return orig["alloc_decode"](seq_lens, seq_lens_cpu, last_loc)

    def available_size():
        L = _lender
        return orig["available_size"]() + (L.lent_pages() * a.page_size if L is not None else 0)

    def clear():
        L = _lender
        if L is not None and L.lent:
            L.take_back(L.lent, "clear")
        r = orig["clear"]()
        global _want_lend
        _want_lend = True
        return r

    a.alloc, a.alloc_extend, a.alloc_decode, a.available_size, a.clear = (
        alloc, alloc_extend, alloc_decode, available_size, clear)


# -- hooks ---------------------------------------------------------------------------

def register_runner(runner) -> None:
    """ModelRunner._init_post_memory_pool_components: every runner's pool and
    allocator (the MTP draft shares the target's allocator)."""
    if not ENABLED:
        return
    if not kt_hot_cache.ENABLED:
        raise RuntimeError("KT_HOT_KV_LEND=1 needs the dynamic hot cache (KT_HOT_DYNAMIC=1)")
    if os.environ.get("KT_HOT_CPU_PREFIX_MASK", "0") == "1":
        # the CPU MoE would skip logical ids below N whatever the tables say
        raise RuntimeError("KT_HOT_KV_LEND=1 with KT_HOT_CPU_PREFIX_MASK=1: residency must be carried by the ids alone")
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

    a = runner.token_to_kv_pool_allocator
    pool = runner.token_to_kv_pool
    _runners.append((bool(getattr(runner, "is_draft_worker", False)), pool, a))
    kvp, ts = _kv_tensors(pool)
    _log(f"{'draft' if _runners[-1][0] else 'target'} pool {type(pool).__name__} on {type(a).__name__} "
         f"(id {id(a) & 0xffff:04x}), {len(ts)} K/V tensors")
    if type(a) is not PagedTokenToKVPoolAllocator:
        _log(f"allocator {type(a).__name__} is not PagedTokenToKVPoolAllocator: nothing lent")
        return
    _wrap_allocator(a)
    cache = kt_hot_cache._cache
    if not _runners[-1][0]:
        if cache is None or not cache.ready:
            _log("the hot cache is not seeded when the pool is built: nothing lent")
            return
        # before the graphs are captured
        _make_tables(cache)


_want_lend = True


def _threshold() -> int:
    from sglang.srt.layers.moe import kt_lend

    t = kt_lend._threshold()
    return t if t > 0 else 1 << 62


def _streamed(fb) -> bool:
    from sglang.srt.layers.moe import kt_lend

    if kt_lend.ENABLED and kt_lend.STATE.get("phase") in ("prefill", "decode"):
        return kt_lend.STATE["phase"] == "prefill"
    return kt_lend._streamed(fb)


def before_forward(runner, fb) -> None:
    """ModelRunner.forward, after the lend's phase switch, outside any graph."""
    global _lender, _want_lend
    if not ENABLED or getattr(runner, "is_draft_worker", False):
        return
    cache = kt_hot_cache._cache
    if cache is None or not cache.ready or not _tables_ready:
        return
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = fb.forward_mode
    if _lender is None:
        a = runner.token_to_kv_pool_allocator
        if not getattr(a, "_kt_kv_lend", False):
            return
        pools = [p for _, p, al in _runners if al is a]
        _lender = Lender(a, pools, cache)
    L = _lender
    if mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
        if _streamed(fb):
            if RELEASE_AT_PREFILL and L.lent:
                STATS["released_at_prefill"] += 1
                L.take_back(L.lent, "streamed prefill")
            _want_lend = True
            return
        L.try_lend("extend")
        _want_lend = True
    elif (mode.is_decode() or mode.is_target_verify()) and _want_lend:
        _want_lend = False
        L.try_lend("decode")
