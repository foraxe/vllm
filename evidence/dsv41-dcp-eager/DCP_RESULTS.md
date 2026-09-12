# DeepSeek V4.1 eager DCP implementation

## Result

The initial explicit-communication DCP path runs the real DeepSeek-V4.1-Flash
model with MRV2 on four GB200 GPUs. TP4/DCP1, TP4/DCP2 and TP4/DCP4 produce identical
tokens on four mixed-batch fixed-answer prompts. A 23,023-token retrieval prompt
also returns identical correct tokens in all three configurations and exercises
candidate pruning beyond the 16K-record candidate budget.

A pinned 32-question, five-shot GSM8K subset scores 31/32 for DCP1 and 32/32 for
both DCP2 and DCP4, with no truncated responses or newly incorrect answers.
This is bounded no-regression evidence, not an accuracy-improvement claim.
Generated explanations differ: only 5/32 complete sequences match the control in
each DCP configuration. No general bitwise-parity, throughput or measured-capacity
claim is made.

## Implementation and scope

- Explicit per-cache DCP placement keeps SWA and C2 circular state replicated;
  main/index caches are sharded by logical record. Unspecified placements preserve
  existing MRV2 block-table behavior. Scheduler accounting and worker placement
  use the same policy.
- C2 ownership is applied after compression, with matching main/index mapping,
  owner-packed prefill bounds and compressed-before-localized decode lengths.
- Candidate source 20 reduces scores over global block IDs, preserves NaN padding
  and newest-block score semantics, and publishes candidate blocks independently
  of its unmasked token TopK. Later index sources mask through global record IDs.
  Explicit score/NaN exchanges are chunked; this is the correctness baseline,
  not the direct peer-memory optimization.
- Attention gathers real TP query heads, filters and compacts owner-local decode
  TopK, handles packed prefill request offsets, and includes replicated SWA only
  once. Raw output/LSE partials merge before applying the sink on the head owner.
  Valid counts normalize native empty-row LSE sentinels. Decode scheduler metadata
  is renewed because local candidate counts can change between index epochs.
- Current startup guards require MRV2, eager NVIDIA FlashMLA, DCP2/4, record
  interleave1, FP8 indexer caches, text-only input, disabled prefix caching,
  PCP1, PP1, no DBO and no speculative decoding. Other combinations fail early.

This extends the validated design in designs/dsv41-dcp/DESIGN.md. It does not port
V4.0's ratio-4/128 compressor or claim the historical V4.0 DCP results from #44573.
No competing open V4.1 DCP implementation was found in the refreshed PR search.

## Validation

| Check | Result |
| --- | --- |
| Candidate/short-indexer and attention helper suites | 29 passed |
| Actual distributed candidate selection, DCP2/4 | 2 passed |
| Mixed replicated/sharded MRV2 slot mapping | 2 passed |
| Compressed record owner and owner-packed prefill kernels | 4 passed |
| Existing record/warmup checks | 7 passed |
| KV cache utilities | 104 passed with one visible GPU; 2 PP cases passed with four visible GPUs |
| Ruff, formatting, mypy and applicable commit hooks | Passed |
| Four-prompt model smoke across DCP1/2/4 | 12/12 correct, exact token matches |
| 23,023-token retrieval across DCP1/2/4 | 3/3 correct, exact token matches |
| GSM8K subset, DCP1/2/4 | 31/32, 32/32, 32/32; zero truncations |

The distributed candidate test covers packed row starts, tails, empty owners,
changing inputs, NaNs and comparison with the incumbent dense selector.
Attention tests cover hole compaction, request/plane isolation, causal bounds,
query-head padding exclusion, empty native LSE and single-sink merge.

The two initial KV-utility failures were GPU-count validation for PP2/4 when only
one GPU was exposed; both passed with the required visibility. An earlier worker
run reached 106 passes before a native teardown fault and is not counted as a
clean run. The root checks above exited cleanly.

The first DCP2 startup rejected the checkpoint's vision capability even with
language-model-only mode selected. The guard now checks whether vision execution
is enabled. The first GSM8K control used max_tokens256 and truncated one response;
those artifacts are retained separately and excluded from the accepted comparison.
The accepted runs all use max_tokens1024.

## Model workload and numerics

Common: official model at dba1be0a40aa45a94ad051997016db3960a90277, TP4 GB200,
MRV2 eager, FlashMLA FP8-DS-MLA main cache, FP8 indexer cache, no prefix cache,
max_num_seqs4, max_num_batched_tokens1024, fixed 512MiB KV budget per worker,
greedy sampling and thinking disabled. JIT warmup/autotune are disabled in the
harness. No timing measurements are promoted.

Smoke: max_model_len8192, two 17-token arithmetic prompts plus 2363/2365-token
retrieval prompts, max_tokens12, all four requests submitted together.

Extended check: max_model_len32768; one 23,023-token retrieval request followed by
32 GSM8K requests. Five training demonstrations and the first 32 test questions
are pinned to openai/grade-school-math revision
3101c7d5072418e28b9008a6636bde82a006892c. The chat/five-shot harness and sample size
are explicit; this is not a full standardized leaderboard evaluation.

DCP2 and DCP4 each match 31/32 final control answers and solve the one question
that control answered incorrectly. Both match only 5/32 complete explanation
sequences. Maximum selected-token logprob differences over matched prefixes are
0.39866 and 0.63411 respectively. Distributed accumulation and selection order
change numerical execution; these results do not establish bitwise equality or
explain every token divergence. Larger accuracy evaluation remains a merge gate
for broader qualification.

## Revisions and runtime

Code implementation: 93216d877e8f2af7ed590da7ae75a8202085cc69.
Feature base: 485421b1c3572597a4cfaece04836a843537435a.
The final branch adds usage documentation and an explicit guard for unqualified
MXFP4 indexer caches; qualified FP8 runtime calculations are unchanged.

Native wheel: 46d2b23ac5047a813ebb082122166e4ae09b5f39, aarch64 CUDA13.0, SHA256
3c784ec38f687033d5921ab77f88fb2ee17429aeb1450043807a6556130c1d1b.
All core C++/CUDA sources match the feature base. Selected FlashMLA pins match;
the only external CMake difference is the unused TML FA4 pin. No ABI shim is used.
The established runtime overlay supplies Torch2.13, FlashInfer0.6.18.post1,
CuTeDSL4.6.2 and DeepGEMM at8b1392b; see native-provenance.json and the prior
runtime-provenance.json. The new source pin requests CuTeDSL4.7, so the evaluated
4.6.2 overlay is recorded explicitly rather than presented as the default install.

## Reproduction and remaining gates

From /workspace/vllm_dsv41, acquire the shared GPU lease before GPU work. Set:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTHONPATH=$PWD/artifacts/dcp:$PWD/runtime-deps/nvidia_cutlass_dsl/dsl_packages:$PWD/runtime-deps:$PWD/vllm-dcp
vllm-dcp/.venv/bin/python artifacts/dcp/model_check.py --dcp 2 --output /tmp/dcp-smoke.json
vllm-dcp/.venv/bin/python artifacts/dcp/eval_gsm8k.py --dcp 2 --output /tmp/dcp-eval.json
```

Use DCP1 and DCP4 for the other configurations. eval_gsm8k.py reads the pinned
subset JSON; the fetch script reproduces it. compare_eval.py verifies prompt
lengths, retrieval tokens and final-answer differences and records logprob data.
Deployment syntax and restrictions are also in
vllm-dcp/docs/serving/context_parallel_deployment.md.

Before broader qualification: larger accuracy/logprob checks, automatic cache
sizing and high-concurrency capacity measurements, then performance with matched
scheduler work and uncontended GPUs. Graphs, PCP, DBO, speculation, prefix caching,
MXFP4 indexer cache and other attention backends remain separate work. Existing
peer-memory optimizations are not enabled in this control implementation.

All GPU jobs completed and the lease was released. The new feature diff requires
its own human line review and relevant test rerun before a draft upstream PR.
