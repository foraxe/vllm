# DeepSeek V4.1 eager DCP validation

Start with DCP_RESULTS.md. Current source is936c40f on upstream eed1f3d.
The runtime-471 directory contains automatic-sizing stress and32-question
model checks with native binaries matching the base and CuTeDSL4.7.1.
The earlier top-level fixed-budget results remain tied to1fb7991 and its
recorded runtime. This bundle makes no serving-speedup or broad capacity claim.

Source: /workspace/vllm_dsv41/vllm-dcp. Artifacts:
/workspace/vllm_dsv41/artifacts/dcp. Run fetch_gsm8k.py to obtain the pinned
subset; source questions are not redistributed. The drivers retain the exact
workspace paths used for qualification. Acquire GPU ownership before rerunning.
The final focused suite has187 passes plus one gated-config fetch failure;
the same metadata assertions pass with the accessible OPT fixture adapter.
