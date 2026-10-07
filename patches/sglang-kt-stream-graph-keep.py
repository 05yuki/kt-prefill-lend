"""The expert streamer keeps a bounded number of CUDA graphs
(KT_STREAM_GRAPH_KEEP, default 8 row counts besides the largest).

_graph_for captures one graph per (rows, group) and kept every one. Full chunks
share one row count, but a prompt's last chunk has its own, so nearly every
request added graphs that were never replayed again. On V4.1 at chunk 6144
(10-06, rank 0) the device memory outside torch's allocator rose 12-14 MiB per
prompt of a new length, and stayed flat when the same prompt came back:
-5139 -> -5041 MiB over eight prompts of 8K-44K tokens after a 114K one. At that
rate a few hundred requests take the room the prefill lend needs to bring the
weights back.

Now the graphs of the largest row count seen stay, and of the others the
KT_STREAM_GRAPH_KEEP most recently used; older ones are dropped after a device
sync (their replays are done). A dropped row count is captured again if it
comes back.

Usage: python sglang-kt-stream-graph-keep.py <sglang tree> [...]
(source/sglang-dsv41, -mimo, -upstream, -qwen38-next-nvidia-v0519)
"""
import sys
from pathlib import Path

MARK = "KT_STREAM_GRAPH_KEEP"


def once(s, old, new, name):
    if s.count(old) != 1:
        raise SystemExit(f"{name}: anchor found {s.count(old)} times:\n{old[:200]}")
    return s.replace(old, new)


if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree> [...]")
for tree in sys.argv[1:]:
    f = Path(tree) / "python/sglang/srt/layers/moe/kt_stream_prefill.py"
    s = f.read_text(encoding="utf-8")
    if MARK in s:
        print(f"already patched {f}")
        continue
    s = once(s, "import traceback\n",
             "import traceback\n\n"
             "# KT_STREAM_GRAPH_KEEP: CUDA graphs kept for row counts other than the\n"
             "# largest (a prompt's last chunk has its own; sglang-kt-stream-graph-keep.py)\n"
             'GRAPH_KEEP = int(os.environ.get("KT_STREAM_GRAPH_KEEP", "8"))\n', f.name)
    s = once(s, "        self._graphs = {}\n",
             "        self._graphs = {}\n"
             "        self._graph_use = {}  # row count -> None, least recently used first\n", f.name)
    s = once(s, "        graph = self._graphs.get(key)\n",
             "        graph = self._graphs.get(key)\n"
             "        self._graph_use.pop(T, None)\n"
             "        self._graph_use[T] = None\n", f.name)
    s = once(s, "            self._graphs[key] = graph\n",
             "            self._graphs[key] = graph\n"
             "            self._trim_graphs()\n", f.name)
    s = once(s, "    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):\n",
             "    def _trim_graphs(self):\n"
             "        \"\"\"Drop the graphs of the least recently used row counts beyond\n"
             "        GRAPH_KEEP, never those of the largest (the full chunk).\"\"\"\n"
             "        top = max(self._graph_use)\n"
             "        old = [t for t in self._graph_use if t != top]\n"
             "        if len(old) <= GRAPH_KEEP:\n"
             "            return\n"
             "        drop = set(old[: len(old) - GRAPH_KEEP])\n"
             "        torch.cuda.synchronize(self.device)\n"
             "        for k in [k for k in self._graphs if k[0] in drop]:\n"
             "            del self._graphs[k]\n"
             "        for t in drop:\n"
             "            del self._graph_use[t]\n"
             "\n"
             "    def _graph_for(self, layer, T: int, gi: int, base: int, n: int, g):\n", f.name)
    f.write_text(s, encoding="utf-8")
    print("patched", f)
