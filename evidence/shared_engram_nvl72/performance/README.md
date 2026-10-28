# DeepSeek-V4.1 shared Engram: 8K/128K/512K local performance follow-up

This published copy includes the analysis scripts, compact trace derivations,
per-request budget rows and profile run reports. The raw prompt token-ID
fixtures remain in the original local workspace because they derive from
local engineering documents; full PyTorch trace files also remain local due
size. Published summaries carry their source paths and SHA256 provenance.

Date: 2026-10-27. Source: `/workspace/vllm_dsv41/vllm-engram-shared-nvl72`
at `25041176ec3ed52022ff3db3d9f01ecaf6b61a4a` (draft PR
<https://github.com/vllm-project/vllm/pull/64082>). Runtime: Torch
2.14.0+cu130, vLLM Python source at that commit with the published cu130
native wheel from `40fe396b60b0fbf312b5ac55dbf3f68a4e12b785`.
Hardware: one four-GPU GB200 host, normal clocks, uncontended leased GPUs.
This report is local; no second NVL72 host was available.

## Decision

For the measured long-context P TP2 + D TP2 configuration, use
`--max-num-batched-tokens 16384` on **P only** and retain `8192` on D. At 512K,
the P gain passed a >=10% gate in both arm orders, with D within the preset
5% regression guardrail. At 128K, P again improved strongly, but one of two
orders missed the D guardrail by 0.3 percentage point; treat D impact as
unsettled there. Raising P to 32768 produced only 6.7% additional P gain in
one matched pair, below the gate, so the sweep stopped.

No Engram gather-kernel change is justified by the profiles. The shared
table remains a memory-saving feature. Budget tuning affects the model's
prefill scheduling and collective behavior independent of table ownership.

## Workloads and validation levels

- P/D used independent TP2, DP1 instances, NixlConnector pull, FP8 KV,
  `FLASHINFER_MLA_SPARSE_DSV41`, MXFP4 indexer, eager mode, prefix cache off,
  one active request, `max_num_batched_tokens` as stated, and 32 forced decode
  tokens. Both roles used the same checkpoint and one shared Engram generation
  during each budget pair.
- 8K profiling used one saved 8192-token retrieval prompt from
  `/workspace/vllm_dsv41/artifacts/decoder-tail-diagnosis/long-eval-prompts.json`.
  The P/D profiler started after one successful unprofiled warmup. These are
  instrumented event inventories, not serving-time measurements.
- 128K and 512K used 131072 or 524288 token IDs constructed by concatenating
  the first four or all sixteen saved 32K engineering-document prompts in
  `/workspace/vllm_dsv41/artifacts/engram-wkv-32k/prompts.json`. The fixture is
  deterministic and exercises the full input length, but is not a natural
  long conversation or a quality evaluation. `max_model_len` was input plus
  512; KV capacity was fixed at 2 GiB/rank for 128K and 8 GiB/rank for 512K.
- All long-context arms admitted the model and completed NIXL handoff with
  the exact requested input and 32 output tokens. Text hashes sometimes
  differed *within the same arm*, including ordinary storage, so these
  fixtures support performance and route checks only. Do not infer model
  quality or bitwise output equivalence from them.

## 8K profile: Engram is small

One profiled request per storage mode, matched prompt IDs and token counts:

| Role | Ordinary HTTP phase | Shared HTTP phase | Ordinary Engram GPU sum | Shared Engram GPU sum |
| --- | ---: | ---: | ---: | ---: |
| P | 263.2 ms | 265.4 ms | 1.238 ms | 0.731 ms |
| D | 8128.5 ms | 7864.0 ms | 0.559 ms | 0.443 ms |

GPU sums include `_engram_lookup_kernel` and the post-WKV kernel across one
rank's trace. They may overlap other streams and are not critical-path or
serving-wall intervals. The two profiled completion hashes differed, while
the unprofiled warmup hashes matched. The earlier fixed-prompt and GSM8K
controls in the base snapshot establish separate bounded correctness evidence.

## 128K and 512K admission

At the original P8K/D8K budget, one warmup and one retained request per
storage mode completed:

| Input | Ordinary retained P / D | Shared retained P / D | Shared/control P / D | Shared HBM used at ready |
| ---: | ---: | ---: | ---: | ---: |
| 131072 | 2.805 / 6.174 s | 2.830 / 6.165 s | 1.009 / 0.999 | about 152 GiB/GPU |
| 524288 | 12.567 / 6.114 s | 12.401 / 6.410 s | 0.987 / 1.048 | about 159 GiB/GPU |

These are one retained sample each, so they show admission and no obvious
large local cost. They do not establish a stable serving ratio.

## 512K profile: collective arrival wait, not table lookup

The P8K and P16K profiles each captured one shared-table, 512K input /
32-output request after warmup. PyTorch profiling inflated P HTTP time, so
use these traces to inspect kernel count and elapsed GPU intervals, not to
replace the unprofiled A/B below.

| P budget | Profiled P HTTP | Rank-0 GPU kernel interval union | Engram GPU sum | Two-shot all-reduce calls/rank |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 17.265 s | 16.943 s | 49.0 ms | 2560 |
| 16384 | 9.254 s | 6.352 s | 53.0 ms | 1280 |

Engram is below 1% of P wall in both profiles. Its lookup call count halves
with larger chunks, but total Engram GPU elapsed does not materially fall.
The large P difference is associated with collective arrival and other
model work. The two-shot kernel has the **same name and grid** in both
traces. Pairing its calls by order across TP ranks gives:

| P budget | Rank-0 minus rank-1 median start | Rank-0 / rank-1 median kernel duration | Median completion skew |
| ---: | ---: | ---: | ---: |
| 8192 | -2511 us | 2518 / 6.8 us | 0.2 us |
| 16384 | +2097 us | 7.0 / 2104 us | -0.1 us |

The rank that arrives early stays in the collective kernel until the peer
arrives; their completions align. This supports a peer-rendezvous wait
interpretation. The source of the per-rank arrival skew was not isolated.
Call-count halving and the changed skew both contribute to the observed P
gain; the trace does not justify assigning the gain to either alone.

## Matched, unprofiled budget controls

For each pair, one owner generation remained live while P and D were stopped
and restarted between arms. P and D used the same checkpoint, 512K or 128K
IDs, 32 output tokens, NIXL pull route, max context, KV capacity and normal
clocks. One or two warmups preceded three or four retained sequential
requests per arm. The total column is the median of each request's P+D HTTP
elapsed seconds, not the sum of separate phase medians.

| Input / order | P8K/D8K P, D, total | P16K/D8K P, D, total | P16/P8 P ratio | D ratio | Total ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512K, 8K then 16K | 11.941, 5.769, 17.710 s | 6.804, 5.485, 12.418 s | 0.570 | 0.951 | 0.701 |
| 512K, 16K then 8K | 12.473, 5.922, 18.395 s | 6.798, 5.780, 12.590 s | 0.545 | 0.976 | 0.684 |
| 128K, 8K then 16K | 2.909, 5.459, 8.314 s | 1.564, 5.749, 7.295 s | 0.537 | 1.053 | 0.877 |
| 128K, 16K then 8K | 2.925, 5.584, 8.519 s | 1.544, 5.727, 7.251 s | 0.528 | 1.026 | 0.851 |

The preset adoption gate was at least 10% lower P median and at most 5%
higher D median. P16K/D8K passed both 512K order checks. At 128K the P gain
passed both, but the D guardrail passed only the reverse order. The measured
P+D total improved in all four pairs under this low-concurrency, sequential
request workload. This is a configuration-specific component of local P/D
serving, not a maximum-QPS or tail-latency claim.

P16K/D16K was also tried at 512K: P improved, but D median was 9.5% slower
than P8K/D8K in that first pair. Holding D at 8K avoided that observed cost.
P32K/D8K improved P only 6.7% over P16K/D8K in one matched pair, below the
10% incremental gate; stop instead of adding a larger-batch production
recommendation without a substantive gain.

## Artifacts and stop boundary

- Raw profiles and scripts:
  `/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/profiles/`,
  `long_context/`, `budget_ab/`, `run_pd_profile.py`, `run_pd_long.py`,
  `run_pd_budget_ab.py`.
- Inspectable summaries:
  `pd_profile_8k_summary.json` SHA256
  `057cd17c5e0d4f757e23d5d9d55fbe17f17f1058da6717794219abdb21278618`,
  `rank_skew_512k.json` SHA256
  `0899ab42976bbb761b25f6b2895ce5a8f74f73613517f66afdf66147c9fcb578`,
  and `pd_budget_summary.json` (regenerate after any new budget arm).
- No GPU clocks, foreign leases, or unrelated sessions were changed. All
  lease records were released after teardown.
- The second-host NVL72 direct read/serving gate remains untested. That is
  the required next hardware validation before a cross-host performance
  claim. No Engram kernel performance PR is warranted from these local
  profiles.

## Follow-up: 1M input and concurrency

The later [one-million-token local P/D screen](one_million/README.md) uses
1,000,000 input tokens and concurrent waves up to eleven requests. It includes
the matched ordinary/shared controls, compact per-request results, raw reports,
reproduction scripts, and a conditional remote-read sensitivity calculation.
It is single-host evidence and does not establish cross-host NVL72 performance.
