"""kt-kernel: one GPU output buffer per depth slot for every batch size.

KExpertsCPUBuffer.get_buffer caches its buffers for every batch size it has
seen (kt-kernel-buffer-cache-all-sizes.patch: a CUDA graph around
submit_forward bakes the pointers, and dropping a size's buffers segfaulted
on replay). The pinned host buffers are small, but output_gpu is
[batch_size, hidden] bf16 x buffer_depth on the GPU: 32 MiB at a 2048-token
prefill chunk, cached once for each chunk size (512..2048 by 256). With the
streamed prefill's per-size buffers that took the margin of Vision-Exp's 1/4
pool, and the scheduler died with CUDA OOM every 20-31 requests (09-29).

Now output_gpu for every size is a view of the head of one base per depth
slot, sized max(batch_size, 2048) rows. sync_forward copies the whole
output_cpu into it before returning it, so no stale rows are read. A view
held in capture_buffers keeps its base alive, so a graph that baked an older
base's pointer still points at live memory after the base grows.

KT_SHARED_GPU_OUTPUT=0 restores the per-size buffers.

Usage: python kt-kernel-shared-gpu-output.py <path to kt_kernel/experts_base.py> [...]
"""
import sys
from pathlib import Path


def once(s, old, new, name):
    if new in s:
        return s
    if s.count(old) != 1:
        raise SystemExit(f"{name}: anchor found {s.count(old)} times:\n{old[:200]}")
    return s.replace(old, new)


for arg in sys.argv[1:]:
    f = Path(arg)
    s = f.read_text(encoding="utf-8")
    s = once(s,
        '''    temp_buffer: tuple = tuple()
    buffer_depth: int = 2
''',
        '''    temp_buffer: tuple = tuple()
    buffer_depth: int = 2
    # (device, dtype, hidden) -> (rows, [base] * buffer_depth); see _gpu_output.
    gpu_output_bases: Dict = dict()

    @classmethod
    def _gpu_output(cls, batch_size, hidden_size, device, dtype):
        """output_gpu for one batch size. With KT_SHARED_GPU_OUTPUT=1 (default)
        every size gets views of one base per depth slot instead of its own
        [batch_size, hidden] tensors; a view keeps its base alive, so graphs
        that baked an older base stay valid when the base grows."""
        if os.environ.get("KT_SHARED_GPU_OUTPUT", "1") != "1":
            return [
                torch.zeros((batch_size, hidden_size), device=device, dtype=dtype)
                for _ in range(cls.buffer_depth)
            ]
        key = (str(device), dtype, hidden_size)
        rows, bases = cls.gpu_output_bases.get(key, (0, None))
        if bases is None or rows < batch_size:
            rows = max(batch_size, 2048)
            bases = [
                torch.zeros((rows, hidden_size), device=device, dtype=dtype)
                for _ in range(cls.buffer_depth)
            ]
            cls.gpu_output_bases[key] = (rows, bases)
        return [b[:batch_size] for b in bases]
''', f.name)
    s = once(s,
        '''        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]
''',
        '''        output_gpu = cls._gpu_output(batch_size, hidden_size, hidden_states.device, hidden_states.dtype)
''', f.name)
    f.write_text(s, encoding="utf-8")
    print(f"patched {f}")
