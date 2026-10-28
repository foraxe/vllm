# DeepSeek-V4.1 shared Engram: local 1M P/D capacity

Captured 2026-10-27 on one four-GPU GB200 host. Production source was vLLM
draft PR #64082 at `46706ee5e5ade3f622b6f606bbd05b84b06a9b96`; Torch
2.14.0+cu130 ran with the published `40fe396b` native wheel. Model config
SHA256: `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`;
index SHA256: `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
All GPUs were leased and uncontended; the lease and processes were released.
This is same-host evidence. Cross-host NVL72/IMEX access was not measured.

## Workload and stop criteria

P TP2 ran on GPUs 0/1, D TP2 on 2/3, both DP1 with NIXL pull, FP8 KV,
`FLASHINFER_MLA_SPARSE_DSV41`, MXFP4 indexer, eager mode and prefix caching
disabled. The prompt was exactly 1,000,000 token IDs: repeat the sixteen saved
32K engineering-document fixtures and truncate. It is a constructed capacity
fixture, not a natural conversation or model-quality test. The model advertises
1,048,576 maximum positions; servers used `max_model_len=1,000,512` and forced
32 decode tokens. P/D budgets were 16,384/8,192 batched tokens, with 24 GiB
fixed KV per rank. In each run, request threads launched together and each
performed P then D through the ordinary NIXL metadata handoff. The first C1
wave warmed the engines. The configured maximum was eight sequences for the
matched storage comparison and eleven for a separate shared-only edge check.

Acceptance required exact 1,000,000 input and 32 output tokens for every
request, NIXL pull metadata, clean P/D/owner shutdown and no GPU lease leak.
The sweep stopped at eleven because P reported 11.59 million KV-token capacity
at 24 GiB and throughput had already plateaued by eight. A twelve-request
wave with this fixed KV reservation would exceed the reported full-context
concurrency estimate, so it was not launched.

## Captured results

Each cell is full P-to-D wave wall seconds / aggregate input tokens per second.
The C1 first wave is startup-warmup and omitted below. C2 has two retained
waves; C4, C8 and C11 each have one.

| Concurrent 1M requests | Ordinary | Shared |
| ---: | ---: | ---: |
| 1 | 22.81 / 43.84K | 22.47 / 44.51K |
| 2, first | 38.20 / 52.36K | 42.57 / 46.98K |
| 2, second | 38.23 / 52.32K | 37.52 / 53.31K |
| 4 | 69.64 / 57.43K | 68.82 / 58.12K |
| 8 | 131.79 / 60.70K | 130.43 / 61.34K |
| 11 | not run | 178.92 / 61.48K |

All 28 shared and 17 ordinary request completions in the retained waves had
the exact requested token counts; all processes exited zero. (Including the
warmup waves: 30 shared and 18 ordinary completions.) Shared C11's first and
last P phases took 17.12 and 172.68 seconds; median D phase was 6.07 seconds.
P completion times in C4/C8 advanced approximately one 1M request every 15-16
seconds. C8 to C11 added only 0.23% aggregate input throughput but increased
the last P phase by about 39%. D ran on separate GPUs and stayed around six
seconds per request, so P scheduling/compute dominated this workload.

The local shared/ordinary C1, C4 and C8 wave rates differ by about 1-2%; C2
varied more between sequential waves. This one-order comparison does not prove
a storage-mode speedup, a tight regression bound, or production SLOs. Ready
GPU memory was 181,059/180,365 MiB on shared P ranks and 175,977 MiB on D
ranks. After waves, shared P rank 0 used 184,477 MiB of 189,471 MiB total:
about 4,994 MiB headroom. P reported 11,592,407 KV tokens at 24 GiB,
corresponding to 11.59 full 1,000,512-token sequences. D reported 13,993,910
KV tokens. Eleven is the tested admission capacity for this exact setup;
roughly four to eight is the useful throughput range before marginal gains
collapse. Latency-sensitive serving may prefer fewer than four.

## Conditional remote Engram estimate

The current GB200 path runs a host lookup inline on the main stream when a
scheduled chunk contains at least 4,096 tokens. Most 16K P chunks meet that
threshold, so their lookup cost is not hidden by earlier-layer compute on the
same stream. Smaller chunks can prefetch on a second stream. Concurrency here
did not turn the large P chunks into parallel completed prefills.

For a full 1M request, the nominal FP8/E8M0 source-read volume per TP rank is
`1,000,000 tokens * 12 local hash columns * (256 + 8) bytes * 2 layers =
6.336 GB`. Actual fabric transactions and cache behavior are unmeasured. A
512K P16K *same-host* trace summed 53 ms for Engram lookup plus its post
kernel on one rank; linear scaling suggests about 0.10 s at 1M, which is a
rough comparator, not a remote measurement. If a remote fabric route delivered
an effective **random-read** rate of 25, 10 or 5 GB/s per rank, the nominal
read times would be 0.25, 0.63 or 1.27 s. Treating every extra lookup second
as critical-path time against the measured ~16 s warmed P phase gives about
1%, 3% or 7% additional P latency, respectively. The same fractions are a
first-order throughput penalty near P saturation. The estimate excludes
unmeasured remote mapping reachability, transaction amplification, cache,
contention across ranks, graph interaction and any software retry. It must
not be presented as observed NVL72 performance.

The decisive next experiment is the same C1/C4/C8 P/D sweep with P and D on
distinct hosts in one NVL72/IMEX domain, with matched ordinary controls and
rank-paired lookup traces. Stop first if direct remote FABRIC import or graph
replay fails; the cross-host gate is documented in
`/workspace/vllm_dsv41/NVL72_TWO_HOST_VALIDATION.md`.

## Reproduction and artifacts

Run from `/workspace/vllm_dsv41/vllm-engram-shared-nvl72` using its `.venv`
and the shared GPU lease for GPUs 0-3. The captured driver was
`/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/run_pd_1m_capacity.py`.
Its default waves are `1,1,2,2,4,8`; `--baseline` selects ordinary tables.
The edge check used `--max-seqs 11 --waves 1,11`. The derivation script is
`summarize_pd_1m_capacity.py`, and its compact output is
`capacity_1m/summary.json`. Full per-request reports and P/D/owner logs stay
under `capacity_1m/`; the compact public evidence copy includes report hashes
and the exact prompt hash without publishing raw token IDs.
