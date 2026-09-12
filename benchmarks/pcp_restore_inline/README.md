# Class-free PCP multicast

Reference: #49756 at5e82de5fc. Inline candidate:f935a4a9b, built on Lucas
Wilkinson's foraxe/vllm#5 at a258551c7. The restorer class is removed; multicast
buffers and the three-operation restore path live in the existing PCP manager.
Allocation remains before KV profiling with coordinated fallback and teardown.
The later main merge does not change this multicast algorithm; no post-merge
engine rerun is claimed.

## Engine comparison

GLM-5.2 NVFP4, 4xGB200, Torch2.13+cu130, eager MRV2, TP1PCP4EP4DCP1, FP8 KV,
FlashInfer CUTLASS MoE, FLASH_ATTN prefill, unchanged sparse decode. C4, each
1024 input/32 greedy output tokens, prefix cache off, chunked prefill on,
model/batch limit4096, max sequences4, 1024 KV blocks.

Both arms share model weights, KV, input and multicast buffers. Implementation
methods and consumer eligibility switch; sampled-row metadata work remains
part of the comparison. Four warmup batches, then six alternating measured
pairs. Full batches queued before scheduling. Each rank executes one prefill
and31 decode steps; every paired generated-token sequence matches.

| Median | Reference | Inline | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 17.9516 tok/s | 17.8666 tok/s | -0.47% |
| TTFT | 276.778 ms | 278.110 ms | +0.48% |
| TPOT | 221.257 ms | 222.085 ms | +0.37% |

This is a bounded preservation check, not a speedup or powered equivalence
claim. The paired throughput geometric mean is -0.35%; a descriptive paired
bootstrap95% interval is [-1.45%,+0.80%]. Initial inline measurements before
group-name caching and lazy dense-index upload are retained separately.

## Components and correctness

BF16/FP16, selected rows1/16/64/256/1024/4096, balanced/single-owner skew.
Both production manager methods use the same multicast allocations. Eight
warmup pairs and30 measured pairs; exact outputs match. Legacy JSON column
`multicast_us` names the reference; `collective_us` names the inline candidate
(it still uses multicast, not AllGather).

This is not a uniform component win: five of24 median points are >5% slower,
maximum+7.45%, while the minimum is-6.38%. BF16 balanced4096 is187.38/187.58us;
skewed4096 is413.68/400.21us. Small-row Python-side overhead remains a caveat.

93 CPU tests pass (two GPU-only cases skipped); four-GPU dense-oracle test
passes, including changing inputs, output lifetime, delayed rank and repeated
release. Allocation/rendezvous failure agreement and lint/mypy pass.

## Reproduction

Use a coherent Torch/native runtime. Prepare a reference worktree at5e82de5fc
and candidate atf935a4a9b. Run:

```bash
PCP_RESTORE_REFERENCE_ROOT=/absolute/path/to/reference \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_NCCL_SYMM_MEM=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 PYTHONPATH=/absolute/path/to/candidate \
.venv/bin/python benchmarks/pcp_restore_inline/engine_ab.py \
  --model /path/to/GLM-5.2-NVFP4 --synchronized-batch --output /tmp/inline-ab.json
```

For the component harness, under PCP_RESTORE_WORKSPACE create worktrees named
`vllm-pcp-restore-lean` at5e82de5fc and `vllm-pcp-multicast-current` atdefd2cbee
(the latter supplies the unchanged historical timing helper). Set
PCP_RESTORE_OUTPUT to an output JSON path and run component_ab.py with the
same four-GPU visibility, disabled optional AllReduce backends, and candidate
PYTHONPATH. Serialization is for trusted local benchmark callbacks only.
