# DeepSeek V4.1 microbatch index-state qualification

Feature: ec241884a7dc7ecbff3db69da47963b9008d6245.
Base: 7ef4d9bfed6311e3b78a40abb4a8bb6a2fc741b0.
Integration: 8276917 (feature plus the eight-file input/history corrections from
PR #56440, head 5a67e49). Companion changes are not in the submitted feature branch.

## Result

- Non-microbatched control: 12/12 expected answers, only microbatch 0 observed.
- DBO: 12/12 expected answers, exact output token IDs match control. Microbatches
  0 and 1 observed on all four EP ranks. Each rank passed 8 top-k checks between
  layers 2 and 3 and 8 candidate-block checks between layers 20 and 24.
- Shared-storage negative control: in the same resident engine, aliasing the
  persistent top-k/candidate buffers back to slot 0 reproduced the top-k overwrite
  on all four EP ranks. Microbatch 0, 1024 rows: 179830, 179831, 179830, 179834
  mismatched entries for ranks 0, 1, 2, 3 respectively. Collision records were
  written before assertions stopped the model. This demonstrates the overwrite;
  it does not establish that it caused the NaNs reported in #56440.

## Configuration and limits

4x GB200, DP2 x TP2 + EP, DeepEP LL, sequence parallelism, text-only, eager,
no prefix cache, synchronous scheduling, 2048 model/scheduled-token limit,
16 max sequences, 1 GiB KV cache per worker. Twelve prompts of about 1218 input
tokens, four fixed-answer questions repeated with a long prefix; greedy, 32-token
output cap. Outputs were 323, Paris, 4 and 100, each repeated three times.

One resident DBO-configured engine runs all phases. The control sets the
microbatch thresholds above the workload; the candidate restores thresholds
2 decode / 16 prefill. Allocation, weights and expert backend remain the same.
The final negative phase changes only persistent index storage ownership.
All checks use eager hooks and synchronize when comparing buffers. These are
correctness diagnostics, not performance measurements or a broad accuracy eval.

NVSHMEM_REMOTE_TRANSPORT=none selects local NVLink transport; NVSHMEM_QP_DEPTH=8192.
The default IBRC path stalled during initialization and is not qualified.
DeepEP d4f41e4, NCCL 2.30.7, NVSHMEM 3.4.5, Torch 2.13; remaining pinned versions
and exact-base native wheel checksum are in artifacts/runtime-provenance.json.
CUDA-graph full-model DBO and inter-node execution are not validated.

An earlier DBO-enabled trial with uneven DP prompt lengths never activated
microbatch 1 and failed the harness activation gate. It is excluded from this
qualification. The negative phase intentionally raises buffer assertions;
subsequent empty-output/engine-shutdown errors are consequences of that phase.

## Reproduce

Acquire all four GPUs through the shared lease registry, then:

```bash
cd /workspace/vllm_dsv41/vllm-dbo-integration
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_DEEP_GEMM_WARMUP=skip \
NVSHMEM_QP_DEPTH=8192 NVSHMEM_REMOTE_TRANSPORT=none \
LD_LIBRARY_PATH=/workspace/vllm_dsv41/dbo-runtime-deps/nvidia/nccl/lib \
PYTHONPATH=/workspace/vllm_dsv41/artifacts:/workspace/vllm_dsv41/dbo-runtime-deps:/workspace/vllm_dsv41/runtime-deps/nvidia_cutlass_dsl/dsl_packages:/workspace/vllm_dsv41/runtime-deps:$PWD \
.venv/bin/python ../artifacts/dbo_model_eval.py --dbo --paired --negative-control \
  --output ../artifacts/dbo-qualified.json
```

Artifacts: dbo-qualified.json, .control.json, .negative.json, per-rank JSONs,
and dbo-qualified.log under artifacts/. The process exited 0 after recording
and catching the expected negative-control collision. All GPU workers exited;
GPU lease released.
