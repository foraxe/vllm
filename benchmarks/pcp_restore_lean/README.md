# PCP restore simplification: performance guard

Reference: published #49756 at `defd2cbee`. Candidate: `076464d21`.
This evidence branch is deliberately separate from the feature diff.

## Final controlled comparison

4xGB200, GLM-5.2 NVFP4, Torch 2.13+cu130, eager MRV2, TP1PCP4EP4DCP1.
FP8 KV, FlashInfer CUTLASS MoE, FLASH_ATTN prefill and the unchanged sparse
decode consumer. Four requests, each 1024 input/32 output tokens, greedy,
prefix cache off, chunked prefill on, 1024 KV blocks. Model/batch limit4096,
max sequences4. Full batch enqueued before scheduling starts.

Four warmup batches; six measured A/B pairs alternating order. The same model,
KV cache, input buffers and multicast allocations are used by both arms;
only manager and restore methods switch. Sampled-row metadata allocation
remains part of the implementation being compared. Each batch executes
one prefill plus31 decode steps on every rank. All generated tokens match.

| Median metric | Published PR | Simplified | Change |
| --- | ---: | ---: | ---: |
| Output throughput | 18.7095 tok/s | 18.8412 tok/s | +0.70% |
| TTFT | 257.330 ms | 258.744 ms | +0.55% |
| TPOT | 212.338 ms | 210.673 ms | -0.78% |

Throughput favors the candidate in four of six pairs. This supports retaining
performance in this bounded workload, not a general speedup or a powered
equivalence guarantee. The initial four-pair comparison used separate manager
input buffers and showed -0.50% throughput/+0.98% TTFT; its raw data is retained
but the shared-buffer comparison is the stronger control.

Component gate: BF16 and FP16, selected rows1/16/64/256/1024/4096, balanced and
single-owner-skewed layouts. Eight warmup pairs and40 measured pairs using the
same allocations and rendezvous handle. 22/24 median points improve; the two
slower points are +1.05% and +0.15%. Exact outputs and previous-output lifetime
pass. Initial separate-allocation timings are retained for transparency.

## Reproduction

Prepare reference and candidate worktrees at the commits above, and use a
coherent Torch/native-extension installation. Run the harness from this branch:

```bash
PCP_RESTORE_REFERENCE_ROOT=/absolute/path/to/reference \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
PYTHONPATH=/absolute/path/to/candidate .venv/bin/python \
benchmarks/pcp_restore_lean/engine_ab.py --model /path/to/GLM-5.2-NVFP4 \
  --synchronized-batch --output /tmp/engine-ab.json

PCP_RESTORE_REFERENCE_ROOT=/absolute/path/to/reference \
PCP_RESTORE_OUTPUT=/tmp/component-ab.json \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
PYTHONPATH=/absolute/path/to/candidate .venv/bin/python \
benchmarks/pcp_restore_lean/component_ab.py
```

The serialization setting is for trusted local benchmark callbacks only.
The engine harness is specifically configured for this GLM-5.2 workload.
The merge of main after076464d21 does not change the PCP restore/sampling code;
55 CPU tests and lint/mypy passed afterward. No post-merge model rerun is claimed.
