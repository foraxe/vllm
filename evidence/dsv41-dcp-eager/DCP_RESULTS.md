# DeepSeek V4.1 eager DCP implementation

## Result

The initial NVIDIA FlashMLA DCP feature is implemented at 936c40f on upstream
eed1f3d, using MRV2 eager execution and FP8 indexer caches. The current-runtime
qualification uses native binaries from that exact upstream base and the source
pin CuTeDSL 4.7.1. TP4/DCP1,2,4 pass all four fixed-answer stress requests,
including two 23K-token retrieval requests. The pinned 32-question five-shot
GSM8K scores are 31/32, 31/32, 32/32; no truncation or newly incorrect answer.

Automatic cache sizing initially underprofiled DCP4 by 3.22 GiB/rank at batch 8192.
The feature now profiles native sparse-attention auxiliaries, query exchange,
FP32 merging, mixed-layout metadata, and small-batch lazy workspaces before
selecting the KV budget. DCP2 and DCP4 stay 151 MiB and 148 MiB below the predicted
budget on the bounded stress workload. The unchanged DCP1 path exceeds its
budget by 448 MiB; that upstream limitation remains explicit. This is a sizing
correctness check at concurrency 4, not a throughput or broad capacity claim.

Earlier fixed 512 MiB KV-budget qualification is retained below with its own
runtime and workload. Those runs produced exact smoke/retrieval tokens across
DCP1/2/4 and scored 31/32, 32/32, 32/32. Neither evaluation establishes general
bitwise parity or an accuracy improvement.

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

## Earlier fixed-budget validation

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
The accepted runs all use max_tokens 1024.

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

## Earlier revisions and runtime

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

The feature diff requires its own human line review and relevant test rerun
before a draft upstream PR.

## Current-runtime automatic-sizing qualification (2026-09-12)

Source 936c40f, base eed1f3d0c6043bd494424a22443ee198dd56f657; all core native
sources and external pins match the wheel. Wheel SHA256:
a542622166880eeb9a18c85629b4008e217c26c14f319a696b5f9201b93b2e4b.
The isolated dcp-runtime-deps overlay uses CuTeDSL 4.7.1 and retains
NumPy 2.3.5/protobuf 6.33.6. Existing runtime-deps remains intact.

Matched settings: TP4, MRV2 eager, maxlen 32768, maxbatch 8192, maxseq 4,
gpu_memory_utilization 0.55, no fixed KV bytes, APC disabled, FP8 indexer,
FlashMLA, seed 0. Stress prompts contain 17, 23023, 23024, 17 tokens and return
323, 654321, 271828, 91 exactly in all arms. Evaluation uses the same pinned
first 32 test questions/first 5 train examples, greedy, thinking False,
max_tokens 1024, without logprob collection. Q12 is the sole incorrect answer
in DCP1/2 (12 instead of 13); DCP4 answers all 32 correctly.

| DCP | GSM8K | Truncated | Runtime peak minus profiled budget, per rank |
| --- | --- | --- | --- |
| 1 | 31/32 | 0 | +448.36MiB (unchanged upstream underprofiling) |
| 2 | 31/32 | 0 | -151.00 to -151.01MiB |
| 4 | 32/32 | 0 | -147.77 to -147.78MiB |

Runtime peak is estimated from free memory after emptying the allocator cache
plus Torch peak-minus-current live allocation. KVCacheConfig descriptors alias
one backing tensor: count that allocation once, not the sum of descriptor sizes.
The acceptance threshold was the requested budget plus 64 MiB; DCP2/4 are below
the budget itself. This workload does not qualify higher concurrency or other
batch sizes. The native collective fallback warnings are retained in raw logs;
no serving performance is claimed.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTHONPATH=$PWD/artifacts/dcp:$PWD/dcp-runtime-deps/nvidia_cutlass_dsl/dsl_packages:$PWD/dcp-runtime-deps:$PWD/runtime-deps:$PWD/vllm-dcp
vllm-dcp/.venv/bin/python artifacts/dcp/runtime_qualification.py --dcp 2 --output artifacts/dcp/runtime-471/final-dcp2-auto.json
```

Use DCP1/4 for the other arms. Raw JSON/logs and exact native provenance are in
artifacts/dcp/runtime-471/. The interrupted pre-native-refresh DCP1 run and
profiling development attempts are retained separately and excluded from the
matched final comparison. Post-merge attention/runner-profiling checks passed 24
cases. Final focused suite: 187 passed; one existing V4 metadata case failed on a
gated Llama config fetch (HTTP401). Its unchanged slot-mapping assertions pass
with the helper selecting the accessible OPT-125m config. No model weights are
loaded by that supplemental check. Both logs and the fixture adapter are retained.
All final feature-file hooks, including mypy, pass.
