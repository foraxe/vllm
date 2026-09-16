# Engram SP shared-memory delivery qualification

Final code: `acbdd4470823fd055bebe2b0340afdd2724812e5`.
Upstream base: `ca542ee154e91c9a7757fb0b8f9d90ee578fb613`.
Worktree: `/workspace/vllm_dsv41/vllm-engram-sp-direct`.

## Decision

Ready for human review and a relevant test rerun before submitting an upstream
draft PR. The user accepted the observed throughput and removed the previous
5% improvement requirement. This is an opt-in execution/storage feature with
bounded performance evidence, not a universal speedup claim.

The final diff adds full-table host visibility for SP token owners, keeps the
existing path disabled-by-default, and includes focused tests and usage docs.
No production logic changed after the initial 0dfb3b0 implementation; the four
production/test file blobs are identical. The branch was rebased and the
feature documentation was added. Native sources/build pins are unchanged
between the 3bb7826 wheel and the ca542ee base; provenance records the comparison.

## Throughput matrix

TP4/EP4, DP1, GB200, MRV2, DeepGEMM mega-MoE, normal CUDA graphs, APC off,
4 GiB KV/rank, max batch tokens 8192,64 forced output tokens per request.
Each case has a warmup and three timed repeats. Prompt hashes and scheduler
signatures match in every pair; no preemptions. These are offline engine batches,
not HTTP serving results or a statistical speedup guarantee.

| Input tokens | Batch | Control tok/s | Direct tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1 | 103.35 | 104.14 | +0.76% |
| 8192 | 1 | 95.68 | 101.33 | +5.90% |
| 512 | 16 | 1466.10 | 1477.10 | +0.75% |
| 512 | 64 | 2908.84 | 2978.69 | +2.40% |
| 8192 | 16 | 347.11 | 348.49 | +0.40% |
| 8192 | 64 | 423.98 | 431.71 | +1.82% |
| 512 / 2048 / 8192 mixed | 24 | 726.68 | 709.73 | -2.33% |

The B1 regression uses the previous synthetic prompt construction and max_num_seqs=4.
The wider matrix uses varied repository-document contexts, max_num_seqs=64 and
graph sizes 1/2/4/8/16/32/64. The mixed batch contains eight requests at each of
512,2048 and 8192 tokens. Compare control/direct within each row; absolute rates
across the two harness configurations are not interchangeable.

Mixed lengths regress 2.33% in this measurement. The option remains off by default.
The prior B4 measurements remain historical evidence in ENGRAM_SP_DIRECT_RESULTS.md,
not another fresh row in this matrix.

## Model-output qualification

Ten fixed-answer arithmetic/retrieval canaries pass and match between modes.
Context retrieval checks pass at 512/8192 tokens. Exact component row parity covers
strided IDs, uneven heads/tokens, empty rank slices, padding and graph replay.

Five-shot GSM8K uses the pinned 32-question subset, source revision
`3101c7d5072418e28b9008a6636bde82a006892c`; prompt/dataset SHA256s are in raw JSON.
The initial 1024-token B1/eval runs scored 31/32 each; direct mode exhausted the
output budget on the same break-even question control answered incorrectly.
That run is retained and is not presented as an untruncated accuracy pass.

The wider matrix used 2048 output tokens: control 32/32, direct 31/32, with no
truncation. A predeclared fixed two repeats per mode, using the same 2048-token
allowance and max_num_seqs=64 configuration, then produced:

| Mode    | Repeat 0 | Repeat 1 | Truncated outputs |
| ------- | -------: | -------: | ----------------: |
| Control |    31/32 |    32/32 |                 0 |
| Direct  |    32/32 |    31/32 |                 0 |

The only varying answer in these repeats is the same break-even question 12
(12 versus 13 years). Both modes show the same score range. This supports treating
the initial one-answer gap as baseline variability in this sample; it is not a
proof of general accuracy equivalence. No repeat-until-pass procedure was used.
Free-form outputs were already variable within control; no bitwise model-output
claim is made.

## Checks, memory and scope

Final rebased code: 120 Engram kernel/distributed tests pass. Applicable pre-commit
hooks and Python 3.12 mypy pass. Full/piecewise model graphs execute. GPU jobs ran
under the shared lease, were serialized, and ended before lease release.
No clocks or global OS policies changed.

Shared physical table size is about 188.83 GiB, one mapping per layer visible to
all TP ranks. It is not four physical copies, and is not a reduction relative
to one ordinary TP-sharded replica. Staging stays 24 MiB/layer/rank. Registration,
NUMA locality and startup costs remain; earlier rank 0 allocation/registration
phases cost about 12 seconds per layer. No startup-speedup or peak-HBM claim.
Existing engine forced-teardown warnings remain in logs.

Enabled scope: CPU-offloaded DSV4.1, same-host TP>1, DP1/PP1, real SP and MRV2;
no CP, microbatching/DBO, speculation, sleep, DP storage modes or threaded loading.
The usage documentation explains the prerequisites and configuration.

## Reproduce and review

Code and tests are in the feature branch. Raw data, logs and scripts are in
`artifacts/engram-sp-direct/`; the exact scripts retain this host's absolute paths
and dependency overlays. `delivery-provenance.json` records library versions.
Acquire GPU ownership before running:

```bash
bash artifacts/engram-sp-direct/reproduce_delivery.sh unit
bash artifacts/engram-sp-direct/reproduce_delivery.sh batch1-control /tmp/b1-control.json
bash artifacts/engram-sp-direct/reproduce_delivery.sh batch1-direct /tmp/b1-direct.json
bash artifacts/engram-sp-direct/reproduce_delivery.sh matrix-control /tmp/matrix-control.json
bash artifacts/engram-sp-direct/reproduce_delivery.sh matrix-direct /tmp/matrix-direct.json
bash artifacts/engram-sp-direct/run_eval_recheck.sh
```

The batch-one script deliberately retains its strict no-truncation assertion;
its original direct evaluation failed that assertion despite matching 31/32 score.
The wider and repeated evaluations record truncations explicitly rather than
hiding them. For a bounded human test rerun, the unit command exercises 120 tests.

The draft PR description is ENGRAM_SP_DIRECT_PR.md. Related open PRs were checked:

PR #56435 exchanges head shards; #56989 packs exchanged rows; #56929 fuses reorder/

quantization; #56926 changes scheduling. This feature removes exchange through
shared visibility and token ownership. No upstream PR/comment has been sent.
