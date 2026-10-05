"""Streamed prefill: one set of graph input buffers for every chunk size.

The graph path kept persistent inputs (xq, sf, ids, w, acc, out) per exact
token count T, and T is the prefill chunk: 512..2048 in steps of the 256-token
page, seven sizes. Every size stayed allocated for the life of the server,
about 90 MB per size per rank together with kt-kernel's per-size output
buffer. The last chunk of each prompt has its own length, so the seven sizes
fill in over the first ~20 requests; on Vision-Exp (2.66 GB free after the
1/4 pool) that ate the margin a long prompt's later chunks need, and the
scheduler died with CUDA OOM every 20-31 serial requests (09-29, corpus).

Now each buffer is a view of the head of one flat base per buffer name,
sized for max(T, 2048) rows. A graph captured for a smaller T bakes the
view's pointer, which is the base's; the view is held in self._gin, so the
base lives as long as the graph. If a larger T ever needs a bigger base, the
older views keep the old base alive; nothing dangles. The groups of a layer
and the layers of a chunk all use one T, so no two live uses share rows.

KT_STREAM_SHARED_INPUTS=0 restores the per-size buffers.

Apply once to source/sglang-dsv41.
"""
import sys
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree>")
root = Path(sys.argv[1])
f = root / "python/sglang/srt/layers/moe/kt_stream_prefill.py"


def once(s, old, new):
    if new in s:
        return s
    if s.count(old) != 1:
        raise SystemExit(f"{f.name}: anchor found {s.count(old)} times:\n{old[:200]}")
    return s.replace(old, new)


s = f.read_text(encoding="utf-8")

s = once(s,
    'ZEROCOPY = os.environ.get("KT_GPU_STREAM_ZEROCOPY") == "1"\n',
    'ZEROCOPY = os.environ.get("KT_GPU_STREAM_ZEROCOPY") == "1"\n'
    '# KT_STREAM_SHARED_INPUTS=1 (default): the graph inputs of every chunk size\n'
    '# are views of one base per buffer (kt-stream-shared-graph-inputs, 09-30).\n'
    'SHARED_INPUTS = os.environ.get("KT_STREAM_SHARED_INPUTS", "1") == "1"\n')

s = once(s,
    '''        self._graphs = {}
        self._gin = {}
''',
    '''        self._graphs = {}
        self._gin = {}
        self._gin_base = {}  # buffer name -> flat uint8 base (SHARED_INPUTS)
''')

s = once(s,
    '''        if g is None:
            g = {
                "xq": xq.clone(),
                "sf": None if sf is None else sf.clone(),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
            }
            self._gin[T] = g
''',
    '''        if g is None and SHARED_INPUTS:
            k = physical_ids.shape[1]
            g = {
                "xq": self._gin_view("xq", T, xq.shape, xq.dtype, x.device),
                "sf": None if sf is None else self._gin_view("sf", T, sf.shape, sf.dtype, x.device),
                "ids": self._gin_view("ids", T, (T, k), torch.int32, x.device),
                "w": self._gin_view("w", T, (T, k), torch.float32, x.device),
                "acc": self._gin_view("acc", T, (T, self.K), torch.bfloat16, x.device),
                "out": self._gin_view("out", T, (T, self.K), torch.bfloat16, x.device),
            }
            self._gin[T] = g
            g["xq"].copy_(xq)
            if sf is not None:
                g["sf"].copy_(sf)
        elif g is None:
            g = {
                "xq": xq.clone(),
                "sf": None if sf is None else sf.clone(),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=x.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=torch.float32, device=x.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=x.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=x.device),
            }
            self._gin[T] = g
''')

s = once(s,
    '''    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
''',
    '''    def _gin_view(self, name, T, shape, dtype, device):
        """A view of the head of the shared base for `name`, grown to
        max(T, 2048) rows' worth of bytes when this T does not fit."""
        numel = 1
        for d in shape:
            numel *= int(d)
        esize = torch.empty((), dtype=dtype).element_size()
        need = numel * esize
        base = self._gin_base.get(name)
        if base is None or base.numel() < need:
            rows = max(T, 2048)
            nbytes = max(need, -(-need * rows // T))
            nbytes = -(-nbytes // 256) * 256
            # Views made from an older base keep it (and their graphs) alive.
            base = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            self._gin_base[name] = base
        return base[:need].view(dtype).view(shape)

    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):
''')

f.write_text(s, encoding="utf-8")
print(f"patched {f}")
