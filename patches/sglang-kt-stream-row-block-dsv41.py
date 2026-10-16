"""The KT streamed prefill of the DeepSeek-V4 lineage (sglang-dsv41 and its copy
sglang-mimo: V4.1, V4-Flash-Vision, MiMo-V2.6-Flash): the CUTLASS MoE runs over
the chunk's rows in blocks (KT_GPU_STREAM_ROW_BLOCK=<rows>, 0 = off), as
sglang-kt-stream-row-block.py does for the NVFP4 / GGUF streamer, so its
workspace is sized for a block, not for the chunk.

At chunk 23,552 MiMo asked 3,091 MB of workspace for the streamer and 2,947 MB
for one call of the resident (hot) experts, about 130 KB a row; at 35,000 rows
the resident call's 4.6 GB per-call allocation ran a planned window out of
memory (10-16). A token's output does not depend on the other rows of a call.

- kt_stream_prefill.py: ROW_BLOCK / _block_rows; the W16 group body (bf16
  activations, no scale factors) calls the MoE through _cutlass_rows, once per
  block of rows inside the group's graph; run() sizes the workspace and the
  tuning bucket for a block. The MXFP8 path keeps its rows (its swizzled scale factors are not cut);
- moe_runner/flashinfer_cutlass.py: a workspace asked for the chunk is asked for
  a block, and the resident experts' MXFP4 call goes over row blocks during a
  KT streamed prefill (each block quantizes its own activations).

Usage: python sglang-kt-stream-row-block-dsv41.py <sglang tree>
"""
import sys
from pathlib import Path

srt = Path(sys.argv[1]) / "python/sglang/srt"


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


patch("layers/moe/kt_stream_prefill.py", [
    ("""W16 = os.environ.get("KT_GPU_STREAM_W16", "0") == "1"
""", """W16 = os.environ.get("KT_GPU_STREAM_W16", "0") == "1"
# sglang-kt-stream-row-block-dsv41.py: rows per CUTLASS MoE call (0 = the chunk);
# the W16 group body and the resident experts' call during a streamed prefill
ROW_BLOCK = int(os.environ.get("KT_GPU_STREAM_ROW_BLOCK", "0") or 0)


def _block_rows(T: int) -> int:
    return min(T, ROW_BLOCK) if ROW_BLOCK > 0 else T
"""),
    # the W16 group body's MoE call goes through _cutlass_rows (row blocks); the
    # call's own arguments are left as the tree has them
    ("""            out.zero_()
            self._cutlass_fused_moe(
                input=xq,  # bf16 activations in this mode
""", """            out.zero_()
            self._cutlass_rows(
                input=xq,  # bf16 activations in this mode
"""),
    ("""    def _group_body(self, layer, gi: int, base: int, n: int, xq, sf, topk_weights, ids32, acc, out):
""", """    def _cutlass_rows(self, **kw):
        \"\"\"sglang-kt-stream-row-block-dsv41.py: the CUTLASS MoE over blocks of
        KT_GPU_STREAM_ROW_BLOCK rows (input, ids, weights and output cut alike).\"\"\"
        rows = kw["input"].shape[0]
        step = _block_rows(rows)
        if step >= rows:
            return self._cutlass_fused_moe(**kw)
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            part = dict(kw)
            for k in ("input", "token_selected_experts", "token_final_scales", "output"):
                part[k] = kw[k][r0:r1]
            self._cutlass_fused_moe(**part)
        return kw["output"]

    def _group_body(self, layer, gi: int, base: int, n: int, xq, sf, topk_weights, ids32, acc, out):
"""),
    ("""        self._tune_tokens = next_power_of_2(T)
""", """        # W16: the group body calls the MoE over blocks of rows
        T_call = _block_rows(T) if W16 else T
        self._tune_tokens = next_power_of_2(T_call)
"""),
    ("""        if self._workspace is None or self._workspace_tokens < T:
""", """        if self._workspace is None or self._workspace_tokens < T_call:
"""),
    ("""            T_ws = _workspace_max_tokens(T)
""", """            T_ws = _workspace_max_tokens(T_call)
            if W16:
                T_ws = _block_rows(T_ws)
"""),
], "ROW_BLOCK")

patch("layers/moe/moe_runner/flashinfer_cutlass.py", [
    ("""    return max(num_tokens, chunk)""",
     """    # sglang-kt-stream-row-block-dsv41.py: a streamed prefill's calls see a block at most
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

    return max(num_tokens, _block_rows(chunk))"""),
    ("""    assert isinstance(quant_info, FlashInferCutlassMxfp4MoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"
    )
""", """    assert isinstance(quant_info, FlashInferCutlassMxfp4MoeQuantInfo), (
        f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"
    )

    # sglang-kt-stream-row-block-dsv41.py: during a KT streamed prefill the
    # resident experts run over blocks of rows (KT_GPU_STREAM_ROW_BLOCK), each
    # quantizing its own activations, so the workspace is a block's
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

    rows = dispatch_output.hidden_states.shape[0]
    step = _block_rows(rows)
    if (step < rows and dispatch_output.hidden_states_scale is None
            and getattr(dispatch_output, "hidden_states_pre_quant", None) is None
            and _kt_streamed_prefill()):
        topk = dispatch_output.topk_output
        if TopKOutputChecker.format_is_bypassed(topk):
            topk = topk.to_standard()
        out = None
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            part_topk = type(topk)(*[
                f[r0:r1] if isinstance(f, torch.Tensor) and f.dim() > 0 and f.shape[0] == rows else f
                for f in topk])
            part = dispatch_output._replace(
                hidden_states=dispatch_output.hidden_states[r0:r1], topk_output=part_topk)
            y = fused_experts_none_to_flashinfer_mxfp4(part, quant_info, runner_config).hidden_states
            if out is None:
                out = torch.empty((rows,) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
            out[r0:r1].copy_(y)
        return StandardCombineInput(hidden_states=out)
"""),
    ("""def get_shared_cutlass_workspace(nbytes: int, device: torch.device) -> Optional[torch.Tensor]:
""", """def _kt_streamed_prefill() -> bool:
    \"\"\"Inside a KT prefill window (the lend's prefill phase).\"\"\"
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return kt_lend.ENABLED and kt_lend.phase() == "prefill"


def get_shared_cutlass_workspace(nbytes: int, device: torch.device) -> Optional[torch.Tensor]:
"""),
], "_kt_streamed_prefill")
