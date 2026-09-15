# DeepSeek V4.1 mega-attention follow-up

Status: two-file captured-query-padding follow-up implemented and validated
for correctness and bounded short-prefill performance. The retained-memory comparison is complete. No new upstream PR has been
submitted; human review and test rerun are the remaining submission gate.

## Which PR is current

- FlashMLA #23 merged the caller-output API; its merge commit credits foraxe.
- vLLM #56344 is closed after its old base branch was deleted.
- #56935 is its replacement, from `zyongye/vllm:v41-megakernel` into upstream
  main. It already contains the shared FP8 buffer and captured output-projection
  design explored in #56544, and it has no split-KV threshold/fallback.
- #56544 remains draft against `vllm-project/vllm:v41-megakernel`. Its description
  now identifies the replacement and labels its old results as historical.
  The identically named branches are in different repositories; an existing
  PR cannot be retargeted to a base repository on another fork.
- A new follow-up should address only a distinct remaining gap on #56935.

Live PR state and source diffs are in `artifacts/fused-refresh/`. No equivalent
query-padding follow-up was found in the recorded upstream/fork searches.

## Source and change

Parent PR head: `7cbfd2049489025046919a8d472584972d3eecc2`.
Its parent/native-wheel revision: `e6eb0d120c480aa5116aafc2d39f19c8ebcf3181`.
The PR changes no native source/dependency files relative to that wheel.
FlashMLA pin: `0eee43b12f034b657133cf2afca6a72ebb6efccf`, including merged #23.
Native provenance and SHA256 are in `artifacts/fused-refresh/native-provenance.json`.

Candidate: `5488ce13f5265fa64d157fd912f34554b991f6c7` on
`perf/dsv41-mega-captured-qpad`, worktree `/workspace/vllm_dsv41/vllm-mega-qpad`.
Two files, 62 additions / 6 deletions, including the regression tests.

```text
ASIS
graph [Q projection, KV insert] -> eager [Q padding, indexer, mega attention]

PR
graph [Q projection, KV insert, Q padding] -> eager [indexer, mega attention]
```

The mega layer prepares padded Q through `_prepare_q_and_insert_kv`, before
MRV2's eager attention break. `forward_mqa` consumes that prepared buffer.
The same padding kernel preserves the fused layout and zero-filled extra
heads. Native attention, quantization, KV record, output buffers and projection
math are unchanged. There is no new model-global scratch buffer or setting.
Full-graph decode already captures this operation, so short-prefill dispatch
is the motivation. The no-padding head count returns the original Q view.

## Validation

- Nine layout tests pass. The new tests change Q between replays, check every
  live fused-layout chunk, and require padded heads to stay zero. Both 16-head
  cases fail on the original PR head and pass with the change. The 64-head
  no-padding case is also covered.
- The exact pinned FlashMLA caller-output suite passes all 31 cases, including
  disjoint prefill/decode segments in shared output buffers.
- Applicable pre-commit hooks, commit hooks, and Python 3.12 mypy pass.
- TP4 GB200, MRV2, FULL_AND_PIECEWISE, explicit `fp8_ds_mla`, FP8 indexer,
  CPU Engram offload. Official model revision
  `dba1be0a40aa45a94ad051997016db3960a90277`.
- Model maximum length 33,792; max sequences 4; max batched tokens 8,192;
  KV allocation fixed at 4 GiB per rank; prefix caching off; graph capture
  sizes `[1,2,4,32,64]`; 64 forced output tokens for timing.
- Every live rank runs 40 mega-attention layers with 16 local heads padded to
  64. Diagnostic replay uses one piecewise and two full graphs. In the final
  control it calls Python query padding eagerly 40 times per worker; candidate
  calls it zero times. Both execute zero eager output-projection calls.
  Startup records padding/projection capture. These diagnostic hooks are
  removed before warmups and timing.

## Matched performance

Control A1, candidate B1, candidate B2, control A2: two separately loaded
engines per arm, two warmups and three measured requests per context/engine.
A separate split-KV investigation ran between A1 and B1. All jobs were serial
under the same four-GPU lease; clocks were left at their normal settings.
These are offline engine results, not an HTTP serving benchmark.

| Input tokens | Control TTFT | Candidate TTFT | Change | Control TPOT | Candidate TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 24.915 ms | 22.743 ms | -8.72% | 5.9305 ms | 5.9009 ms |
| 8192 | 171.162 ms | 168.891 ms | -1.33% | 6.5136 ms | 6.5025 ms |

Short-prefill per-engine medians: A1 25.007, B1 22.600, B2 22.963,
A2 24.823 ms. Both order comparisons pass the >=5% improvement gate.
TPOT changes stay within 0.51%; no decode speedup is claimed.

The 8K result is order-sensitive: A1/B1 is 176.202/167.490 ms;
A2/B2 is 165.979/169.015 ms. Do not promote the pooled 8K median as a gain.
No larger workload sweep was performed.

Prompt hashes, 64-token output lengths, and scheduler-step signatures match
across all four engines. No preemption occurs. All 20 short-context completions
(including warmups) are token-identical. Long-context generations are not
bitwise stable: control alone has 10 distinct token sequences across its 10
requests, candidate has six; only one of ten paired completions matches.

Mixed scheduling uses 17- and 16,384-token prompts and 32 outputs each. Its
first steps are `[(0,17),(0,8175)]`, `[(17,1),(8175,8191)]`, then
`[(18,1),(16366,18)]`, covering decode plus prefill and non-aligned boundaries.
All four scheduler signatures match; the four runs contain two distinct output
pairs. This is successful mixed execution, not global token equality.

## Accuracy and default format

GSM8K first 32 test questions with five fixed training examples, greedy,
thinking disabled, max output 1,024 tokens:

| Run | Correct | Incorrect question IDs |
| --- | ---: | --- |
| Control A1 | 31/32 | 12 |
| Candidate B1 | 32/32 | none |
| Candidate B2 | 32/32 | none |
| Control A2 | 32/32 | none |

No answer reaches the output cap. Full generated strings vary even between
controls (10/32 token sequences match) and between candidates (11/32 match).
The small canary shows no new incorrect answers; it does not establish broad
accuracy, a quality improvement, or model-level bitwise equivalence.
The original plan's wording about unchanged accuracy outcomes is interpreted
as a non-regression gate, not identical generated wording: the control itself
changes from 31/32 to 32/32 across loads.

A separate candidate run with explicit `nvfp4_ds_mla` (the backend default)
passes arithmetic and mixed execution, reports the expected format on every
rank, and scores 31/32 on the same canary without truncation. No matched NVFP4
performance or accuracy-equivalence claim is made.

The separate split-KV control scores 32/32. Its TPOT is 5.390 ms at 17 input
tokens and 5.923 ms at 8K, versus about 5.93/6.51 ms for unmodified mega.
This single-load comparison is directional and indicates that capturing Q
padding does not justify recommending the mega backend for general TP4 decode.

## Limits and reproduction

MRV1, DBO, speculative decoding, other TP sizes and non-GB200 devices are not
qualified by this experiment. Startup graph-capture logs show 0.21 GiB per rank for both implementations.
A separate matched memory check after mixed and 8K prefill measured about
83.1103 GiB retained PyTorch allocation in both arms. Reserved memory is
86.1875 GiB control versus 86.1895 GiB candidate (+2 MiB/rank). The first
mixed run peaks at 85.2214 versus 85.5333 GiB (+0.312 GiB/rank); the later
8K run peaks at 85.2213 versus 85.2247 GiB (+0.0034 GiB/rank). Moving padding
earlier can overlap more temporary storage, but this check finds no large
retained allocation increase. These are PyTorch allocator measurements with
fixed KV allocation, not an automatic KV-capacity qualification. Exact values
for every rank are in `artifacts/fused-refresh/memory-summary.json`.

Every engine exits with status 0 and GPUs become empty. The common engine
shutdown path logs worker timeout/forced teardown and a shared-memory cleanup
warning; no clean-shutdown improvement is claimed.

Acquire the shared GPU lease first, then run:

```bash
bash artifacts/fused-refresh/reproduce.sh tests
bash artifacts/fused-refresh/reproduce.sh native
bash artifacts/fused-refresh/reproduce.sh candidate /tmp/qpad-review
bash artifacts/fused-refresh/reproduce.sh control /tmp/qpad-control
bash artifacts/fused-refresh/reproduce.sh lint
```

Use separate output directories and control/candidate/candidate/control order
for timing. `mega_bench.py` and `summarize.py` retain raw output tokens,
prompt/scheduler hashes, per-request metrics, source provenance and diagnostics.
Raw run names map A1/B1/B2/A2 to `mega-fp8-first`, `qpad-fp8-c1`,
`qpad-fp8-c2`, `mega-fp8-b2`. Native and focused test logs are adjacent.

Human review of the new two-file diff and a relevant test rerun are required
before submitting a follow-up PR, per the target repository's AGENTS.md.

The four-GPU lease was released after all model processes stopped. No clock
settings were changed. Review diff:
<https://github.com/foraxe/vllm/compare/7cbfd2049489025046919a8d472584972d3eecc2...5488ce13f5265fa64d157fd912f34554b991f6c7>
