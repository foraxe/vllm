export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DEEP_GEMM_WARMUP=skip
export PYTHONPATH=/workspace/vllm_dsv41/decoder-tail-runtime:/workspace/vllm_dsv41/artifacts/decoder-tail:/workspace/vllm_dsv41/artifacts/engram-sp-direct:/workspace/vllm_dsv41/engram-runtime-deps:/workspace/vllm_dsv41/dcp-runtime-deps/nvidia_cutlass_dsl/dsl_packages:/workspace/vllm_dsv41/dcp-runtime-deps:/workspace/vllm_dsv41/runtime-deps:/workspace/vllm_dsv41/vllm-decoder-tail
