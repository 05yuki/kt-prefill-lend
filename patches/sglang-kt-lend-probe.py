"""Run the prefill-lend probe (kt_lend_probe.py) before the scheduler's event
loop when KT_PREFILL_LEND_PROBE is set: one extend forward per chunk and
prefix over an uncomputed prefix, so the lend records what a chunk costs at
a long prefix without the prefill. Copies kt_lend_probe.py (next to this
file) into layers/moe/.

Usage: python sglang-kt-lend-probe.py <sglang tree> [...]   (source/sglang-dsv41)
"""
import shutil
import sys
from pathlib import Path

MARK = "kt_lend_probe"
OLD = '''def dispatch_event_loop(scheduler: Scheduler):
'''
NEW = '''def dispatch_event_loop(scheduler: Scheduler):
    if os.environ.get("KT_PREFILL_LEND_PROBE"):
        # patches/sglang-kt-lend-probe.py: measure the lend before serving
        from sglang.srt.layers.moe import kt_lend_probe

        kt_lend_probe.run(scheduler)
'''
SRC = Path(__file__).resolve().parent.parent / "kt_lend_probe.py"

if len(sys.argv) < 2:
    raise SystemExit(f"usage: python {Path(__file__).name} <sglang tree> [...]")
for tree in sys.argv[1:]:
    srt = Path(tree) / "python/sglang/srt"
    shutil.copyfile(SRC, srt / "layers/moe/kt_lend_probe.py")
    f = srt / "managers/scheduler.py"
    s = f.read_text()
    if MARK in s:
        print(f"already patched {f}")
        continue
    if s.count(OLD) != 1:
        raise SystemExit(f"{f}: anchor found {s.count(OLD)} times")
    f.write_text(s.replace(OLD, NEW))
    print("patched", f)
