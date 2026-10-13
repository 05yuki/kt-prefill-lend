# GLM-5.3-Flash, as two patches

The serving tree for GLM-5.3-Flash (`profiles/glm53-flash.json`): fp8 dense
checkpoint, NVFP4 experts beside it, KT's NVFP4 CPU method, NVFP4 hot
experts on the GPU, the full-GPU prefill above the KT token threshold with
the lend, fp8 KV, no MTP. Two cards (TP2) as measured: 16-19 tok/s on
Japanese fiction at 131K context.

```
git clone https://github.com/kvcache-ai/sglang.git && cd sglang
git checkout 541ddc37cbc92c60dc748db5ff1a2aad0b069a80
git apply --whitespace=nowarn <repo>/trees/glm53/sglang-541ddc3-glm53.patch
```

`sglang-541ddc3-glm53.patch` (42 files on kvcache-ai/sglang `541ddc3`): the
GLM-5.3 NVFP4 MoE on the GPU (`glm53_nvfp4_moe.py`, the NVFP4 -> fp8 fold),
the fused Sinkhorn router kernel (`glm53_sinkhorn.py`, the decode's largest
fixed cost otherwise), the NSA top-k in torch, the NextN draft model file,
the lend (`kt_lend.py`, `kt_pinned.py`, `kt_launch_timing.py`), the KT
wrapper and model-runner hooks, two Triton MoE configs for the 5070 Ti.

```
git clone https://github.com/kvcache-ai/ktransformers.git && cd ktransformers
git checkout f7607c0f66c643220ff43b822b24483c98c41392
git submodule update --init third_party/llamafile third_party/pybind11
git apply --whitespace=nowarn <repo>/trees/glm53/kt-kernel-f7607c0-glm53.patch
```

`kt-kernel-f7607c0-glm53.patch` (4 files on upstream `f7607c0`): the MXFP4
/ NVFP4 expert kernel with the 1-byte scale codes (`KT_MXFP4_SCALE_CODES`),
the SwiGLU clamp (`KT_SWIGLU_CLAMP`), the task queue spin and `cpuinfer.h`
changes. Build as in `../kt-kernel/README.md` (the Flash-Next build flags),
install the `.so` next to this tree's `kt-kernel/python` copied as the
package.

Both reproduce the serving trees file for file on clean worktrees
(2026-10-13). The serve: `tools/serve.py --profile glm53-flash --venv ... --tree
... --config-dir <fp8 dense checkpoint> --experts <nvfp4 checkpoint> --tp 2
[--placement <frequency placement .pt>]`.

Checked 2026-10-13 through `serve.py`: fired up, 19.4-19.5 tok/s on the
Japanese prompts (the launcher's 18.6). The profile pins hot 10: the plan's
11 left a 34,880-token pool against 85,248 at 10 — GLM's fixed VRAM (the DSA
indexer's K pool, the full-GPU prefill workspace) is about 0.5 GiB above the
planner's Flash-Next constants, so for GLM the measured hot wins over the plan.
