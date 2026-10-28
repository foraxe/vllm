# DeepSeek-V4.1 shared Engram: P/D sequence-cap screen

Captured 2026-10-28 on one leased four-GPU GB200 host. Source was draft
vLLM PR #64082 at `46706ee5e5ade3f622b6f606bbd05b84b06a9b96` with
Torch 2.14.0+cu130 and the published `40fe396b` native wheel. P TP2 used
GPUs 0/1, D TP2 GPUs 2/3, one shared 188.833 GiB Engram owner, NIXL pull,
FP8 KV, eager mode, prefix cache disabled, `FLASHINFER_MLA_SPARSE_DSV41`,
MXFP4 indexer and lazy safetensors. All arms kept a 1,000,512-token model
limit, P/D budgets of 16,384/8,192 tokens, 24 GiB KV per rank, and 32 forced
decode tokens. No second NVL72 host was available. All owner, P/D and client
processes exited normally and the GPU lease was released.

## Question and stop gate

Does setting `--max-num-seqs 128` on our P TP2 and D TP2 engines degrade
long-context P/D serving or improve useful concurrency? A >10% repeated
regression in P or D phase, an admission failure, or a material HBM loss
would trigger source/profile investigation. If the larger cap helped only D,
test P8/D128 once to localize the configuration choice. Stop after a
reverse-order matched comparison and role-split confirmation.

The 1M prompt was exactly 1,000,000 IDs made by repeating saved 32K
engineering-document fixtures. The 8K prompt was one saved retrieval input;
both are fixed token-ID fixtures rather than natural-traffic or quality
evaluations. Every response was checked for exact input/output token counts
and NIXL pull metadata. Client waves used simultaneous threads, each sending
P then D. Wave wall includes both phases and queuing.

## 1M request: no cap-128 regression

The first C1 wave warmed each engine. The table shows the retained C1 and
single C4/C8 waves, in seconds for the complete P-to-D wave.

| P cap / D cap | C1 | C4 | C8 | P/D ready GPU MiB |
| --- | ---: | ---: | ---: | --- |
| 8 / 8 | 22.466 | 68.819 | 130.429 | P 181059/180365; D 175977/175977 |
| 128 / 128 | 22.493 | 68.598 | 128.596 | P 181117/180423; D 176037/176037 |
| 8 / 128 | 22.248 | 68.916 | 130.457 | P 181059/180365; D 176037/176037 |

Cap 128 added about 58-60 MiB per rank at ready, and the measured wave
differences are within 1.5%. P's scheduler still logged one running request
and waiting clients on these 1M waves. The larger cap did not create 128
active long prefills or improve 1M throughput in this configuration.

## 8K request waves: D cap 8 is the bottleneck

Both storage settings here used the same shared Engram owner. The sequence
cap was tested in A/B and reverse B/A order, with a warmup C1 in each arm.
The table reports complete P-to-D wave wall seconds. The role-split arm was
run once after the matched pairs.

| P cap / D cap | C64 waves | C128 waves | C128 median D phase |
| --- | --- | --- | ---: |
| 8 / 8 | 49.583, 51.765, 45.168 | 96.676, 96.178 | 45.621, 44.729 s |
| 128 / 128 | 12.620, 15.170, 12.128 | 22.250, 22.715 | 7.107, 8.014 s |
| 8 / 128 | 14.750, 11.969 | 20.631 | 7.000 s |

At C128, cap 128 cut complete-wave wall by 76-77% in both arm orders and
raised observed input throughput from about 10.8-10.9K to 46.2-47.1K
tokens/s. The role-split arm reached 50.8K tokens/s in one wave, consistent
with D cap being the lever; its apparent advantage over both-128 is not a
stable optimization claim. P median latencies were in the same few-second
range at both caps. P's sampled logs showed at most three running requests
in every 8K arm. D's sampled maximum was eight with cap 8 and 64 with cap
128; D had up to 111 waiting with cap 8. The scheduler source sets the
RUNNING admission limit from `max_num_seqs` unless `max_num_active_seqs` is
separately set. This accounts for the long D queue at cap 8. No Engram kernel
change follows from this result.

## Decision and limits

For this text-only, eager, no-prefix-cache, no-DSpark setup, use **P cap 8,
D cap 128** as the measured configuration. It preserves the 1M C1/C4/C8
behavior and avoids the short-request D queue. Both-128 also passed and was
not materially worse at 1M. The P cap should be revisited for other prompt
mixes, larger P batching, CUDA Graphs or DSpark; our 8K and 1M fixtures do
not establish a universal maximum concurrency or production tail SLO. The
intermediate D cap 64 was not tested, so 128 is a validated setting rather
than a proven minimum for this throughput. The
source PR remains a memory-sharing change, with no code-level performance
patch justified by this screen. Cross-host NVL72 remains untested.

Reproduce from `/workspace/vllm_dsv41/vllm-engram-shared-nvl72` with its
`.venv` and a four-GPU lease using
`/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/run_pd_1m_capacity.py`.
The default source mode is shared. For example, the role-split high-load arm
used `--prefill-max-seqs 8 --decode-max-seqs 128 --prompt-tokens 8192
--waves 1,64,128,64`; the long arm used `--prompt-tokens 1000000
--waves 1,1,4,8`. The compact summary is
`/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/capacity_1m/seqcap_128_summary.json`;
raw reports and P/D logs are in the same `capacity_1m/` directory. The
summary records their SHA256 values and derives scheduler maxima from the
logs. The public evidence copy contains the script, summary, raw JSON and
selected scheduler logs without publishing prompt token IDs.
