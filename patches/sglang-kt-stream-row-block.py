"""KT streamed prefill (NVFP4 and GGUF editions): the CUTLASS MoE of each expert group runs
over the chunk's rows in blocks (KT_GPU_STREAM_ROW_BLOCK=<rows>, 0 = off), so its
workspace is sized for a block, not for the chunk.

The workspace is the bulk of the chunk-sized lend scratch: 1,926 MB for 18,432
rows on Flash-Next (104.5 KB a row of the 114), resumed at every streamed
prefill and growing with the chunk. Each group's experts are still copied in
once per chunk; only the GEMMs are split, all inside the group's graph. A
token's output does not depend on the other rows of the call.

- kt_stream_prefill.py: _group_body loops over row blocks; run() sizes the
  workspace and the tuning bucket for a block.
- quantization/modelopt_quant.py: the workspace preallocated in the lend
  scratch is sized for min(chunk, block);
- moe_runner/flashinfer_cutlass.py: the workspace size asked for during a
  prefill rounds up to a block, not to the chunk, and the resident (hot)
  experts' call runs over row blocks during a KT streamed prefill too (one
  call over 35K rows asked a 3.5 GB workspace outside the lend's, 10-15);
- kt_stream_prefill_gguf.py: the GGUF edition's bf16 MoE in the same blocks.

Usage: python sglang-kt-stream-row-block.py <sglang tree>
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
    ("""def _chunk_rows() -> int:
""", """# sglang-kt-stream-row-block.py: rows per CUTLASS MoE call inside a group (0 = the chunk)
ROW_BLOCK = int(os.environ.get("KT_GPU_STREAM_ROW_BLOCK", "0") or 0)


def _block_rows(T: int) -> int:
    return min(T, ROW_BLOCK) if ROW_BLOCK > 0 else T


def _chunk_rows() -> int:
"""),
    ("""        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
""", """        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        T = x.shape[0]
        R = _block_rows(T)
        for r0 in range(0, T, R):
            r1 = min(T, r0 + R)
            self._moe_rows(slot, g, g1, g2, n, x[r0:r1], local[r0:r1], topk_weights[r0:r1], out[r0:r1])
        acc.add_(out)

    def _moe_rows(self, slot, g, g1, g2, n, x, local, topk_weights, out):
        \"\"\"One CUTLASS MoE call over a block of rows (sglang-kt-stream-row-block.py).\"\"\"
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
"""),
    ("""            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
        acc.add_(out)
""", """            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
"""),
    ("""        self._tune_tokens = next_power_of_2(T)
        if self._workspace is None or self._workspace_tokens < T:
""", """        Tb = _block_rows(T)
        self._tune_tokens = next_power_of_2(Tb)
        if self._workspace is None or self._workspace_tokens < Tb:
"""),
    ("""            nbytes = fp4_workspace_nbytes(
                T, self.K, self.N, self.G * layer.moe_ep_size, physical_ids.shape[1],""",
     """            nbytes = fp4_workspace_nbytes(
                Tb, self.K, self.N, self.G * layer.moe_ep_size, physical_ids.shape[1],"""),
    ("""            self._workspace_tokens = T
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {T} tokens x {self.G} experts")
""", """            self._workspace_tokens = Tb
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {Tb} tokens x {self.G} experts"
                 + (f" (chunk {T} in blocks)" if Tb < T else ""))
"""),
], "ROW_BLOCK")

patch("layers/moe/moe_runner/flashinfer_cutlass.py", [(
    """        chunk = get_global_server_args().chunked_prefill_size or 0
    except Exception:
        chunk = 0
    return max(num_tokens, chunk)""",
    """        chunk = get_global_server_args().chunked_prefill_size or 0
    except Exception:
        chunk = 0
    # sglang-kt-stream-row-block.py: the streamer's calls see a block of the chunk at most
    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

    return max(num_tokens, _block_rows(chunk))""")], "_block_rows")

patch("layers/quantization/modelopt_quant.py", [(
    """                    big = fp4_workspace_nbytes(
                        get_global_server_args().chunked_prefill_size, w2.shape[1], w2.shape[2] * 2,""",
    """                    from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

                    big = fp4_workspace_nbytes(
                        _block_rows(get_global_server_args().chunked_prefill_size), w2.shape[1], w2.shape[2] * 2,"""),
    ("""                    _fc._lent_cutlass_workspace[key] = kt_lend.scratch_empty((big,), torch.uint8, w2.device,
                                                                             per_chunk=True)""",
     """                    # in blocks it no longer grows with the chunk (the chunk hint's per-row cost)
                    _chunk = get_global_server_args().chunked_prefill_size
                    _fc._lent_cutlass_workspace[key] = kt_lend.scratch_empty(
                        (big,), torch.uint8, w2.device, per_chunk=_block_rows(_chunk) >= _chunk)""")],
    "_block_rows")

patch("layers/moe/moe_runner/flashinfer_cutlass.py", [
    ("""    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    quant_scales = quant_info.quant_scales
    workspace_kwargs = {}
""", """    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    quant_scales = quant_info.quant_scales
    workspace_kwargs = {}
    # sglang-kt-stream-row-block.py: during a KT streamed prefill the resident
    # experts' call goes over row blocks as the streamer's does, so the lent
    # workspace sized for a block serves it (a scale-factor input keeps its rows)
    rows_per_call = x.shape[0]
    if x_sf is None and not enable_alltoall and _kt_lend_phase() == (True, "prefill"):
        from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

        rows_per_call = _block_rows(x.shape[0])
"""),
    ("""        nbytes = fp4_workspace_nbytes(
            x.shape[0], w2_weight.shape[1], w2_weight.shape[2] * 16,""",
     """        nbytes = fp4_workspace_nbytes(
            rows_per_call, w2_weight.shape[1], w2_weight.shape[2] * 16,"""),
    ("""    output = flashinfer_cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight,
        fc2_expert_weights=w2_weight,
        output_dtype=output_dtype,
        input_sf=x_sf,
        quant_scales=quant_scales,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        activation_type=_activation_type(runner_config),
        enable_alltoall=enable_alltoall,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
        **workspace_kwargs,
    )[0]
""", """    ids32 = topk_ids.to(torch.int)

    def _call(r0, r1):
        return flashinfer_cutlass_fused_moe(
            output=output[r0:r1],
            input=x[r0:r1],
            token_selected_experts=ids32[r0:r1],
            token_final_scales=topk_weights[r0:r1],
            fc1_expert_weights=w13_weight,
            fc2_expert_weights=w2_weight,
            output_dtype=output_dtype,
            input_sf=x_sf,
            quant_scales=quant_scales,
            ep_size=quant_info.moe_ep_size,
            ep_rank=quant_info.moe_ep_rank,
            tp_size=quant_info.moe_tp_size,
            tp_rank=quant_info.moe_tp_rank,
            tune_max_num_tokens=next_power_of_2(rows_per_call),
            activation_type=_activation_type(runner_config),
            enable_alltoall=enable_alltoall,
            use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
            **workspace_kwargs,
        )[0]

    if rows_per_call < x.shape[0]:
        for r0 in range(0, x.shape[0], rows_per_call):
            _call(r0, min(x.shape[0], r0 + rows_per_call))
    else:
        output = _call(0, x.shape[0])
"""),
], "rows_per_call")

# the GGUF edition of the streamer (kt_stream_prefill_gguf.py, bf16 CUTLASS MoE
# over the dequantized group): the same blocks, its workspace sized for one
patch("layers/moe/kt_stream_prefill_gguf.py", [
    ("""        out.zero_()
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=self.w13_bf16[:n],
            fc2_expert_weights=self.w2_bf16[:n],
""", """        out.zero_()
        from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

        T = x.shape[0]
        R = _block_rows(T)
        for r0 in range(0, T, R):
            r1 = min(T, r0 + R)
            self._moe_rows(n, x[r0:r1], local[r0:r1], topk_weights[r0:r1], out[r0:r1])
        acc.add_(out)

    def _moe_rows(self, n, x, local, topk_weights, out):
        \"\"\"One bf16 CUTLASS MoE call over a block of rows (sglang-kt-stream-row-block.py).\"\"\"
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=self.w13_bf16[:n],
            fc2_expert_weights=self.w2_bf16[:n],
"""),
    ("""            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
        acc.add_(out)

    def _reference_check(self, method, layer_idx, x, ids, w, acc):
""", """            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )

    def _reference_check(self, method, layer_idx, x, ids, w, acc):
"""),
    ("""        self._tune_tokens = next_power_of_2(T)
        if self._workspace is None or self._workspace_tokens < T:
            from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
""", """        from sglang.srt.layers.moe.kt_stream_prefill import _block_rows

        Tb = _block_rows(T)
        self._tune_tokens = next_power_of_2(Tb)
        if self._workspace is None or self._workspace_tokens < Tb:
            from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
"""),
    ("""            T_ws = _workspace_max_tokens(T)
""", """            T_ws = _workspace_max_tokens(Tb)
"""),
], "_moe_rows")
