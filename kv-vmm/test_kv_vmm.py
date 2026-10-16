"""kv_vmm on its own: tensors made in the region read and write like any other,
a released range gives its memory back to the driver (and to torch), the rest
of the tensor keeps its bytes, a restored range reads zero, a CUDA graph
captured before the release reads the restored memory.

  KT_KV_VMM=1 PYTHONPATH=<dir of kt_kv_vmm.py> python test_kv_vmm.py
"""
import torch

import kt_kv_vmm as kv

assert kv.ENABLED, "set KT_KV_VMM=1"
dev = torch.cuda.current_device()
mib = 1 << 20
with kv.region():
    a = torch.empty(256 * mib, dtype=torch.uint8, device="cuda")
    b = torch.empty(64 * mib, dtype=torch.uint8, device="cuda")
chunk = kv.chunk_bytes(dev)
print(f"chunk {chunk >> 20} MiB, a at {a.data_ptr():#x}, b at {b.data_ptr():#x}")
a.copy_(torch.arange(a.numel(), device="cuda", dtype=torch.int64).to(torch.uint8))
b.fill_(7)
ref = a.clone()

# a graph that sums the top half of a, captured before any release
out = torch.zeros((), dtype=torch.int64, device="cuda")
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    out.copy_(a[128 * mib:].sum(dtype=torch.int64))
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out.copy_(a[128 * mib:].sum(dtype=torch.int64))
g.replay()
torch.cuda.synchronize()
before_sum = int(out)

torch.cuda.synchronize()
free0 = torch.cuda.mem_get_info()[0]
n = kv.release(a, 128 * mib, 128 * mib)
free1 = torch.cuda.mem_get_info()[0]
print(f"released {n >> 20} MiB, driver free {free0 >> 20} -> {free1 >> 20} MiB, counted {kv.released_bytes(dev) >> 20} MiB")
assert n == 128 * mib and free1 - free0 >= n - 2 * mib

# torch can use the memory now
x = torch.empty(120 * mib, dtype=torch.uint8, device="cuda")
x.fill_(1)
torch.cuda.synchronize()
assert torch.equal(a[:128 * mib], ref[:128 * mib]), "bottom half of a changed"
assert int(b.sum(dtype=torch.int64)) == 7 * b.numel(), "b changed"
del x
torch.cuda.empty_cache()

r = kv.restore(a, 128 * mib, 128 * mib)
torch.cuda.synchronize()
print(f"restored {r >> 20} MiB, counted {kv.released_bytes(dev) >> 20} MiB")
assert r == 128 * mib and kv.released_bytes(dev) == 0
assert int(a[128 * mib:].count_nonzero()) == 0, "restored range not zero"
assert torch.equal(a[:128 * mib], ref[:128 * mib])
a[128 * mib:].copy_(ref[128 * mib:])
g.replay()
torch.cuda.synchronize()
assert int(out) == before_sum, f"graph read {int(out)} after restore, {before_sum} before"

# a range that is not chunk-aligned only gives back whole chunks
n = kv.release(a, chunk // 2, 3 * chunk)
assert n == 2 * chunk, n
assert kv.restore(a, 0, a.numel()) == 2 * chunk
torch.cuda.synchronize()
assert int(a[chunk:3 * chunk].count_nonzero()) == 0
assert torch.equal(a[:chunk], ref[:chunk]) and torch.equal(a[3 * chunk:], ref[3 * chunk:])
del a, b, g
torch.cuda.synchronize()
print("ok")
