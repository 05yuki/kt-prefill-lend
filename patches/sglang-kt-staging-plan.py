"""KT's shared staging buffer (the CPU experts' copy of a batch, kt_ep_wrapper.py)
under the prefill plan (KT_PREFILL_PLAN=1): sized for the batches it stages
(under the GPU prefill threshold; without a GPU prefill, the lend's own chunk
KT_PREFILL_PLAN_SAFE) rather than the plan's cap, and a batch larger than it
(a multimodal or fallback extend that takes the hybrid route past the GPU
prefill) gets a buffer of its own for that call instead of an assertion.

The buffer is chunk x hidden in the model dtype, made before the KV pool is
sized, so the cap took it out of the KV pool: GLM-5.3-Flash at cap 36,864 had
175,808 full tokens instead of 217,408 (10-16). A planned chunk past the lend's
own runs the GPU prefill, which does not stage.

Usage: python sglang-kt-staging-plan.py <sglang tree>
"""
import sys
from pathlib import Path

f = Path(sys.argv[1]) / "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
s = f.read_text()
if "_gpu = int(self.gpu_prefill_token_threshold" in s:
    print(f"already patched {f}")
    raise SystemExit(0)
pairs = [
    ("""        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192
""", """        self._staging_buffer_max_size: int = kt_config.chunked_prefill_size or 8192
        # sglang-kt-staging-plan.py: under the prefill plan the launch chunk is its
        # cap. Only a batch under the GPU prefill threshold is staged (a larger one
        # that takes the hybrid route gets a buffer of its own), so the buffer is
        # that size; without a GPU prefill, the lend's own chunk (KT_PREFILL_PLAN_SAFE)
        if os.environ.get("KT_PREFILL_PLAN", "0") == "1":
            _safe = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
            _gpu = int(self.gpu_prefill_token_threshold or 0)
            _want = _gpu if _gpu > 0 else _safe
            if 0 < _want < self._staging_buffer_max_size:
                self._staging_buffer_max_size = _want
"""),
    ("""        assert num_tokens <= self.max_tokens, (
            f"Batch size {num_tokens} exceeds staging buffer max size {self.max_tokens}"
        )
        return self.buffer[:num_tokens]
""", """        if num_tokens > self.max_tokens:
            # a batch past the buffer (sglang-kt-staging-plan.py): its own buffer,
            # held until the next call so the CPU stream's copy outlives this one
            self._oversize = torch.empty(
                (num_tokens, self.hidden_size), dtype=self.buffer.dtype, device=self.buffer.device
            )
            return self._oversize
        self._oversize = None
        return self.buffer[:num_tokens]
"""),
]
for old, new in pairs:
    if s.count(old) != 1:
        raise SystemExit(f"{f}: anchor found {s.count(old)} times: {old[:70]!r}")
    s = s.replace(old, new)
if "\nimport os\n" not in s:
    raise SystemExit(f"{f}: no 'import os'")
f.write_text(s)
print(f"patched {f}")
