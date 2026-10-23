"""Install the KV pool on remappable memory (KT_KV_VMM=1, kt_kv_vmm.py over
libkv_vmm.so) and the prefill's loan of the free KV pages
(KT_PREFILL_KV_LEND=1, kt_kv_prefill_lend.py) into a tree that has the KT
prefill lend (kt_lend.py, the unified one).

- utils/torch_memory_saver_adapter.py: the no-op adapter's region("kv_cache")
  becomes kt_kv_vmm.region() when KT_KV_VMM=1 (every pool sglang makes inside
  that region: the K/V buffers, and also the req-to-token and Mamba pools,
  which are never released);
- the loan itself is in kt_lend.py (patches/kt_lend.py, unified 10-15): a window
  that opens lends the free KV pages after its free memory is measured, and
  end_window maps them back before the room check and the weights;
- managers/scheduler.py: every queued request's prompt length, by rid;
- managers/schedule_batch.py: the chunk-cache SWA eviction counts the chunk in
  flight as the one planned for the request (kt_prefill_plan.chunk_in_flight);
- mem_cache/kv_cache_configurator.py: a hybrid SWA model's sliding-window pool
  sized for the largest planned chunk and its free pages lent outside prefill
  (KT_PREFILL_SWA_LEND=1, KT_PREFILL_SWA_TOKENS).

Build the library first (runtime/kv-vmm/libkv_vmm.so, see kv_vmm.c).
Usage: python sglang-kt-kv-vmm.py <sglang tree>
"""
import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
srt = Path(sys.argv[1]) / "python/sglang/srt"
for m in ("kt_kv_vmm.py", "kt_kv_prefill_lend.py", "kt_prefill_plan.py", "kt_kv_hot_lend.py"):
    shutil.copy(here / m, srt / "layers/moe" / m)
    print(f"installed {srt / 'layers/moe' / m}")
if (here / "libkv_vmm.so").exists():  # built beside the patcher: next to the module that loads it
    shutil.copy(here / "libkv_vmm.so", srt / "layers/moe/libkv_vmm.so")
    print(f"installed {srt / 'layers/moe/libkv_vmm.so'}")


def patch(rel, pairs, mark):
    f = srt / rel
    s = f.read_text()
    if mark in s:
        print(f"already patched {f}")
        return
    for old, new in pairs:
        if s.count(old) != 1:
            raise SystemExit(f"{f}: anchor found {s.count(old)} times: {old[:70]!r}")
        s = s.replace(old, new)
    f.write_text(s)
    print(f"patched {f}")


patch("utils/torch_memory_saver_adapter.py", [(
    """class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield
""", """class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        # KT_KV_VMM=1 (sglang-kt-kv-vmm.py): the KV pool on remappable memory
        if tag == "kv_cache":
            from sglang.srt.layers.moe import kt_kv_vmm

            if kt_kv_vmm.ENABLED:
                with kt_kv_vmm.region():
                    yield
                return
        yield
""")], "kt_kv_vmm")

patch("managers/scheduler.py", [(
    """    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
""", """    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        # the prompt length for the prefill's loan of free KV pages (sglang-kt-kv-vmm.py;
        # first, so trees that check the request in different orders take it alike)
        from sglang.srt.layers.moe import kt_kv_prefill_lend

        kt_kv_prefill_lend.note_request(req.rid, len(req.origin_input_ids))
""")], "kt_kv_prefill_lend")

patch("managers/scheduler.py", [(
    """            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size
""", """            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size
        # the chunk planned per request from the room a prefill has, the free KV
        # pages included (sglang-kt-kv-vmm.py, KT_PREFILL_PLAN)
        from sglang.srt.layers.moe import kt_prefill_plan

        chunked_prefill_size = kt_prefill_plan.chunk_for(self, chunked_prefill_size)
""")], "kt_prefill_plan")

# trees whose pool sizing has no configurator (GLM-5.3's) have no SWA pool to lend.
# The configurator: the memory budget at the lend's own chunk (swa_sizing), the
# SWA pool, and a pool that follows its pages (DeepSeek V4's c4 compress state),
# widened for the plan's cap (swa_size), its free pages lent after setup (swa_init)
OLD_SWA = """            if not self.is_draft_worker:
                # KT_PREFILL_SWA_LEND (sglang-kt-kv-vmm.py): the sliding-window pool
                # holds the largest planned chunk; its free pages leave outside prefill
                from sglang.srt.layers.moe import kt_kv_prefill_lend

                swa_max_total_num_tokens = kt_kv_prefill_lend.swa_tokens(swa_max_total_num_tokens)
"""
cfg = srt / "mem_cache/kv_cache_configurator.py"
if cfg.exists():
    text = cfg.read_text()
    if "kt_kv_prefill_lend.swa_size(" not in text and OLD_SWA in text:
        cfg.write_text(text.replace(OLD_SWA, ""))  # the 10-16 morning form
        print(f"took the earlier SWA hunk out of {cfg}")
    patch("mem_cache/kv_cache_configurator.py", [(
        """            config = self._resolve_memory_pool_config(pre_model_load_memory)
""", """            # KT_PREFILL_SWA_LEND (sglang-kt-kv-vmm.py): budgeted at the lend's own chunk
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            with kt_kv_prefill_lend.swa_sizing():
                config = self._resolve_memory_pool_config(pre_model_load_memory)
"""), (
        """        return _PoolSizes(
""", """        if swa_max_total_num_tokens is not None and not self.is_draft_worker:
            # KT_PREFILL_SWA_LEND (sglang-kt-kv-vmm.py): the sliding-window pool holds
            # the largest planned chunk, its free pages leave outside prefill
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            swa_max_total_num_tokens, c4_state_pool_size = kt_kv_prefill_lend.swa_size(
                swa_max_total_num_tokens, self.pool_page_size, c4_state_pool_size
            )

        return _PoolSizes(
""")], "kt_kv_prefill_lend.swa_size(")
    patch("mem_cache/kv_cache_configurator.py", [(
        """        pools = self._init_pools(
            sizes=sizes,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )
""", """        pools = self._init_pools(
            sizes=sizes,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        )
        if self.is_hybrid_swa and not self.is_draft_worker:
            from sglang.srt.layers.moe import kt_kv_prefill_lend

            kt_kv_prefill_lend.swa_init(pools.token_to_kv_pool_allocator, pools.token_to_kv_pool)
""")], "kt_kv_prefill_lend.swa_init(")

# the chunk-cache SWA eviction under the overlap scheduler (trees that have it)
sb = srt / "managers/schedule_batch.py"
if "                                pre_len - get_schedule().chunked_prefill_size\n" in sb.read_text():
    patch("managers/schedule_batch.py", [(
        """                            pre_len = (
                                pre_len - get_schedule().chunked_prefill_size
                                if get_schedule().chunked_prefill_size > 0
                                else pre_len
                            )
""", """                            # the chunk in flight is the one planned for this request
                            # (sglang-kt-kv-vmm.py, kt_prefill_plan), not the launch chunk
                            from sglang.srt.layers.moe import kt_prefill_plan

                            _in_flight = kt_prefill_plan.chunk_in_flight(
                                req, get_schedule().chunked_prefill_size
                            )
                            pre_len = pre_len - _in_flight if _in_flight > 0 else pre_len
""")], "kt_prefill_plan.chunk_in_flight")
