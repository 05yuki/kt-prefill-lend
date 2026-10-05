"""kt-kernel: the CPU experts' GPU output buffer of a streamed prefill in the
KT prefill lend scratch (patches/kt_lend.py), not resident.

experts_base._gpu_output shares one base per (device, dtype, hidden) across
batch sizes (patches/kt-kernel-shared-gpu-output.py, 09-30) and grows it to
the largest batch seen: a prefill chunk of 8192 rows left 2 x 64 MiB per base
resident through decode. Under the lend, Vision-Exp rank 0 kept 0.6 GB outside
the scratch after a 114K-token prefill (this buffer the largest part) and with
16 hot experts per layer the weights could not come back.

While kt_lend is in a prefill window the buffer comes from a second base made
inside kt_lend.scratch_region(); decode and short (unstreamed) prefills keep
the resident base. Scratch contents are not kept between windows, which this
buffer does not need (written before every read).

Also handles a kt-kernel without the shared output (one buffer per batch size):
a buffer made inside a prefill window goes in the scratch.

Usage: python kt-kernel-lend-gpu-output.py <experts_base.py> [...]
"""
import sys
from pathlib import Path

OLD = '''        key = (str(device), dtype, hidden_size)
        rows, bases = cls.gpu_output_bases.get(key, (0, None))
        if bases is None or rows < batch_size:
            rows = max(batch_size, 2048)
            bases = [
                torch.zeros((rows, hidden_size), device=device, dtype=dtype)
                for _ in range(cls.buffer_depth)
            ]
            cls.gpu_output_bases[key] = (rows, bases)
        return [b[:batch_size] for b in bases]
'''
NEW = '''        key = (str(device), dtype, hidden_size)
        # patches/kt-kernel-lend-gpu-output.py: inside a KT prefill lend window
        # the buffer lives in the lend scratch, so the prefill-sized base is not
        # kept through decode
        region = None
        try:
            from sglang.srt.layers.moe import kt_lend

            if kt_lend.ENABLED and kt_lend.phase() == "prefill":
                key = key + ("prefill",)
                region = kt_lend.scratch_region
        except ImportError:
            pass
        rows, bases = cls.gpu_output_bases.get(key, (0, None))
        if bases is None or rows < batch_size:
            rows = max(batch_size, 2048)
            if region is None:
                bases = [
                    torch.zeros((rows, hidden_size), device=device, dtype=dtype)
                    for _ in range(cls.buffer_depth)
                ]
            else:
                with region():
                    bases = [
                        torch.zeros((rows, hidden_size), device=device, dtype=dtype)
                        for _ in range(cls.buffer_depth)
                    ]
            cls.gpu_output_bases[key] = (rows, bases)
        return [b[:batch_size] for b in bases]
'''

# a kt-kernel without the shared output (venv-mimo, venv-qwen38-...): one
# output per batch size, cached by get_buffer; a prefill-window one goes in
# the scratch
OLD_PER_BATCH = '''        output_gpu = [
            torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
            for _ in range(cls.buffer_depth)
        ]
'''
NEW_PER_BATCH = '''        # patches/kt-kernel-lend-gpu-output.py: inside a KT prefill lend window
        # the buffer lives in the lend scratch
        import contextlib

        region = contextlib.nullcontext
        try:
            from sglang.srt.layers.moe import kt_lend

            if kt_lend.ENABLED and kt_lend.phase() == "prefill":
                region = kt_lend.scratch_region
        except ImportError:
            pass
        with region():
            output_gpu = [
                torch.zeros((batch_size, hidden_size), device=hidden_states.device, dtype=hidden_states.dtype)
                for _ in range(cls.buffer_depth)
            ]
'''

for arg in sys.argv[1:]:
    p = Path(arg)
    s = p.read_text()
    if "kt-kernel-lend-gpu-output" in s:
        print(p, "already patched")
        continue
    if s.count(OLD) == 1:
        p.write_text(s.replace(OLD, NEW))
    elif s.count(OLD_PER_BATCH) == 1:
        p.write_text(s.replace(OLD_PER_BATCH, NEW_PER_BATCH))
    else:
        raise SystemExit(f"{p}: neither the shared nor the per-batch output anchor found")
    print("patched", p)
