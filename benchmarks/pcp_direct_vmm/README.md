# Direct VMM PCP graph validation

This harness accompanies vllm-project/vllm#57115. It is a bounded correctness
check, not a performance benchmark. The branch includes the tested feature
commit 2e2c2f3e4b on main ebac032b92.

On four peer-accessible CUDA GPUs with a supported model checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_PCP_DIRECT_KV=1 \
VLLM_GPU_SYNC_CHECK=error \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
PYTHONPATH=. .venv/bin/python benchmarks/pcp_direct_vmm/validate_engine.py \
    --model /path/to/GLM-5.2-NVFP4 \
    --output pcp_direct_vmm_smoke.json
```

The serialization opt-in is used only for trusted local inspection callbacks and
the explicit native-oracle mode switch. It is not a feature or deployment
requirement. Run this harness locally with trusted inputs.

The fixed configuration is MRV2, TP1/PCP4/EP4/DCP1, FP8 KV, native extensible
storage and default LBNHC, FlashInfer sparse MLA with FlashAttention prefill,
FlashInfer CUTLASS MoE, maximum model length128, maximum batch tokens64,
maximum requests4 and gpu_memory_utilization0.85. Compilation is disabled;
FULL_DECODE_ONLY graphs capture sizes1/2/4. Prefix caching is disabled.

It verifies the direct domain borrows the native owner, graph replay actually
occurs, epochs advance and KV/peer-table addresses remain stable. It generates
four greedy tokens from64-token prompts at batches1/2/4/1, then checks token
parity against the native collective eager route in the same engine.

The public PR's pytest suites cover BF16, FP16, FP8 and fp8_ds_mla stores,
PCP4 and TP2/PCP2 group isolation, negative slots, graph replay after physical
growth/remapping and unequal padded segments. Full-model validation was run on
four GB200 with GLM-5.2 NVFP4 weights and Torch2.14.0+cu130.
