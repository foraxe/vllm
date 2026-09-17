# Decoder-tail regression diagnosis and attempted fixes

**Current result:** the fixed-tail design wins on large scheduled prefills. Current-main TP4/EP4 GB200 tests show about 35.6% higher throughput at 8K/B64 with a 32K token budget. Small batches now bypass compaction. See the final current-main section below; earlier negative results are retained to explain the design change.

Follow-up to DECODER_TAIL_RESULTS.md. Source starts at1557801. All measurements
use TP4/EP4/DP1 on the same four leased GB200 GPUs, MRV2, bundled DeepGEMM2.8
with Mega-mHC, the same native wheel and model. No public changes or PR.

## Why the original throughput fell

The original compacts and redistributes mHC/hidden states at all19 suffix layers,
rebuilds private attention metadata, uploads small index arrays and changes SP
collective/kernel shapes repeatedly. The row-count reduction did not eliminate
most eager Python launches. CUDA event spans around collectives include waits
for other ranks and delayed launches; they are not pure network-transfer time.

The original paired8K results imply additional wall time per input request:
B1 +41.05ms, B16 +49.48ms, B64 +52.61ms. B1 spends more of its total run decoding,
so that penalty translates to a6.46% throughput loss; B64 is more prefill-heavy,
so the same approximate per-request penalty becomes27.69%.

A same-loaded-model full/tail/tail/full diagnostic with four8K prompts confirms
about190ms extra CPU work around the suffix versus directly running its layers,
or approximately48ms per prefill. State exchange costs61.65ms CPU across the
four prompts, metadata44.27ms, and uploads21.91ms (uploads overlap those categories;
do not add inclusive ranges). This closely matches the original per-request loss.

Smaller attention shapes genuinely reduce device work. In the finer fixed-tail
trace, suffix attention's CUDA-event span falls103.15ms to42.42ms while its CPU
span remains180.55ms versus177.27ms. The suffix MoE event span only falls83.14ms
to69.77ms: fewer rows do not imply proportionally less expert/communication work.
Collective spans absorb launch gaps/rank waiting, so the saved attention work
does not translate into equivalent end-to-end savings. These diagnostic numbers
include event/annotation overhead and are not stand-alone throughput measurements.

## Attempted fixes

1. Keep one2414-row dependency halo for the entire suffix, or compact once every
   four layers; retain larger intermediate regions instead of redistributing
   every layer. The final128 rows and every layer's cache tail have the same
   complete dependency coverage as the shrinking-halo design.
2. Reuse private metadata by original metadata identity while the compact layout
   remains unchanged. Reset the cache on every model forward or layout change.
   Reuse query/position/padding views and compact shared TopK buffers only once.
3. Round the retained halo up to2560 rows to avoid the original2414-row padding
   branch and use aligned GEMM/SP dimensions. Extra rows are actual earlier prompt
   rows, capped at each request's available query length, not invented tokens.
4. Test forced NCCL in the full model, because isolated collective performance
   cannot predict behavior with inter-rank CPU launch skew. It did not remove
   the remaining loss; keep default collective dispatch.
5. Test running all19 suffix layers in one prepared context. It did not improve
   the result enough to justify the extra implementation; the block executor is
   archived as block-experiment.patch and removed from the final code.

The first uninstrumented diagnostic uses four8K prompts and four forced output
tokens each, two warmups and three measured runs. All prompt hashes and scheduler
signatures match. Full controls average546.58ms and559.80ms; shrinking every layer
747.21ms; every four layers660.92ms; once593.68ms. Thus one-time compaction recovers
most of the original regression but remains slower than the full path.

These fixes preserve dependency coverage, not bitwise equality of arbitrary
floating-point operations after GEMM/MoE shape changes. No whole-model bitwise
identity claim is made.

## Validation and final engine comparison

The CPU nonlinear dependency/cache oracle passes12 tests including chunked
prefill, decode continuation, progressive/staged/fixed tails and aligned halos.
The final model matrix and bounded model evaluation are recorded in
artifacts/decoder-tail-diagnosis/final-matrix.json. Results are appended below
when complete. No HTTP or broad accuracy claim is made.

## Reproduction and evidence

- artifacts/decoder-tail-diagnosis/paired.json: original layer-event comparison.
- fixed-paired.json: fixed-tail subcomponent event comparison.
- ablation.json: compaction-interval timings.
- dispatch-ablation.json: aligned shapes and default versus forced NCCL.
- block-ablation.json: rejected one-scope suffix executor.
- final-matrix.json and final-summary.json: final paired64-output engine cases.
- CPU test and hook logs, exact per-run source patches and benchmark scripts
  are preserved alongside those files.

Source the existing artifacts/decoder-tail/env.sh, prepend
artifacts/decoder-tail-diagnosis to PYTHONPATH, and run its ablate.py with
--matrix --output PATH using vllm-decoder-tail/.venv/bin/python. The script
switches full/fixed-tail modes in the same loaded model, warms each workload,
checks matched scheduler signatures offline, and saves raw results incrementally.

## Final matched 8K-budget results

Local commit: `fa707f96c23f5d96c2a2741cac31c9f2389f7ee4`. The captured source patch matches this commit exactly.

Same loaded model, normal graphs, 64 forced output tokens per request, two warmups and three measured repetitions per arm. All prompt hashes and scheduler signatures match, with no preemptions.

| Input | Batch | Full prefill tok/s | Fixed tail tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 1 | 107.55 | 106.14 | -1.31% |
| 8192 | 16 | 384.75 | 360.47 | -6.31% |
| 8192 | 64 | 465.77 | 436.47 | -6.29% |
| Mixed 512/2048/8192 | 24 | 779.82 | 737.89 | -5.38% |
| 512 | 16 | 1573.99 | 1561.62 | -0.79% |

The original B64 loss was 27.69%; it is now 6.29%. This is a recovery from the prototype regression, not a win over full prefill. Short 512-token requests now retain the full path throughout the suffix. The fixed 2560-row halo reduces suffix rows by 68.75% for an 8192-token request, versus 84.48% for the original shrinking halo.

Eight arithmetic canaries, 8192/8193-token retrieval with decode continuation, and prompt-logprob fallback passed. The fixed five-shot GSM8K subset scores 32/32 with no truncated outputs; the earlier full-control score was 31/32 and is reused here, not rerun. This small evaluation is not an accuracy improvement claim. Many short evaluation requests stay on the full path under the fixed-halo design. Pre-commit and Python 3.12 mypy passed.

## Final current-main feature candidate

Base `5efbf0886a7110ae0801b9540e3072e89a23eb36`, candidate `dec8e010e05e00dd920aa941e740f8e99176be3b`. Six-file review diff: https://github.com/foraxe/vllm/compare/5efbf0886a7110ae0801b9540e3072e89a23eb36...dec8e010e05e00dd920aa941e740f8e99176be3b. The measured model-code patch matches the final commit exactly.

The final version uses a fixed 2560-row halo, reuses metadata, bypasses batches below 16384 scheduled tokens, and caches the no-trim decision for short-only batches. It is opt-in and uses default SP collective dispatch.

| Prompt | Batch | Full prefill tok/s | Fixed tail tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 1 | 107.05 | 106.58 | -0.44% |
| 8192 | 16 | 518.31 | 662.40 | +27.80% |
| 8192 | 64 | 645.93 | 875.96 | +35.61% |
| Mixed 512/2048/8192 | 24 | 1026.10 | 1197.35 | +16.69% |
| 512 | 16 | 1596.31 | 1594.10 | -0.14% |

Both arms use a 32768-token budget, not an 8192-token control. B64 uses ABBA with two warmups and three samples in each arm occurrence (six samples per mode); the other cases use two warmups and three samples per mode. All prompt hashes and scheduler signatures match, with no preemptions. Small B1 batches record zero compaction calls. These are offline-engine results, not HTTP serving or a general latency guarantee.

Validation: 12 CPU dependency/cache tests, applicable hooks and Python 3.12 mypy pass. On current main the short five-shot GSM8K subset scores 31/32; that mostly exercises the full path. The separate paired long-context qualification pads eight pinned questions to exactly 8192 tokens: control 8/8, candidate 8/8, no truncation, and candidate compaction calls increase. Four simultaneous 8192-token retrieval requests pass. The same large batch with prompt_logprobs returns all 8192 entries per request with finite target-token logprobs, schedules at least 16384 tokens together, and records no additional compaction calls.

The performance harness completed its matrix and short qualification, then failed while rendering an oversized temporary prompt for the new long evaluation. No long-evaluation inference ran in that attempt. The corrected standalone qualify.py passes and records pr-qualification.json; its max model length is 10240 to allow 8192 input plus a 2048-token output ceiling. The throughput matrix uses max model length 9216 and 64 forced outputs. Existing shutdown warnings remain.

Runtime provenance: Torch 2.13.0+cu130, coherent bundled DeepGEMM 2.8.0 package extracted from wheel 3bb7826 (ad1f172 pin), not the older external 2.6.1 fallback. NVIDIA csrc and CMakeLists are unchanged between that native wheel and current main. The current-main default DeepGEMM pin is a6bbb80; the tested external package is explicitly pinned to the coherent older 2.8 package in both arms. Current main removes redundant MoE intermediate padding, and this fresh comparison includes that change. Missing vendored FlashMLA/CuTe Python helpers in the initial new worktree were restored before the accepted run. Shared Python environments were not modified.

### Rerun on this workspace

```bash
cd /workspace/vllm_dsv41
source artifacts/decoder-tail-diagnosis/env-pr.sh
vllm-decoder-tail-pr/.venv/bin/python -m pytest vllm-decoder-tail-pr/tests/models/test_deepseek_v41_decoder_tail.py -q
# Reserve the shared GPUs before either model run.
vllm-decoder-tail-pr/.venv/bin/python artifacts/decoder-tail-diagnosis/qualify.py
vllm-decoder-tail-pr/.venv/bin/python artifacts/decoder-tail-diagnosis/ablate.py --matrix --confirm --batch-tokens 32768 --output /tmp/dsv41-tail-rerun.json
```

No upstream PR has been opened. The user authorized a draft on a performance win; repository AGENTS.md additionally requires human review of every changed line and a relevant test rerun before submission. The source branch and draft body are prepared for that final review. All GPU jobs ended and the lease was released.
