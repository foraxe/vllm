# DeepSeek V4.1 sparse-index ownership validation

See DBO_MODEL_RESULTS.md for scope, results, and exact reproduction configuration.
The integration branch is validation/dsv41-dbo-current on foraxe/vllm. It adds
PR #56440 input/history corrections to the feature branch for model qualification.

The harness requires local DeepSeek-V4.1-Flash weights. Adapt the model and
dependency paths to the host. Run from the integration checkout; use a fresh
output prefix for each invocation. It intentionally terminates the engine after
recording the expected shared-storage collision. This is a correctness diagnostic,
not a throughput benchmark or general model-accuracy evaluation.

The JSON files preserve successful control/DBO outputs and the subsequent negative
control separately. Per-rank collision records were written before assertions.
manifest.json supplies checksums. No full-model CUDA-graph or inter-node claim.

The exact executed harness is preserved as dbo_model_eval.executed.py.txt.
The .py copy differs only in formatting, import sorting, and SPDX headers.
