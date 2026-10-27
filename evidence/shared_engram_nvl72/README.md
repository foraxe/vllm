# DeepSeek-V4.1 shared Engram: local four-GB200 evidence

This folder mirrors the local baseline snapshot. Captured reproduction scripts
are archived verbatim as `.py.txt` files in [`scripts/`](scripts/); copy one to
`.py` before executing it in the documented workspace. Small raw result JSON is
in [`results/`](results/), and
selected execution logs in [`logs/`](logs/). Absolute `/workspace` paths below
record the original capture location; `MANIFEST.json` hashes this published
copy. Synthetic token strings in component scripts are test fixtures, not
production credentials.

Date: 2026-10-27. Source worktree:
`/workspace/vllm_dsv41/vllm-engram-shared-nvl72`, commit
`25041176` (implementation at `c5ecd3b1`, plus a TP-layout test commit),
rebased on vLLM main
`a9b727a6122a0817ecb4eba61baeeb52e9a7d7f5`.
Draft PR: <https://github.com/vllm-project/vllm/pull/64082> .
RFC: <https://github.com/vllm-project/vllm/issues/64002> .

## Provenance and workload

- Hardware: one four-GPU GB200 host, two Grace NUMA nodes, NVIDIA driver
  580.105.08; no second NVL72 host was available. All GPU runs used the shared
  lease file `/workspace/baselines/worker_state/gpu_leases.json`; leases were
  released after each run.
- Runtime: isolated `.venv`, Torch 2.14.0+cu130, CUDA bindings 13.4.4, vLLM
  Python source at the commit above. Native cu130 binaries came from the
  published precompiled wheel for `40fe396b60b0fbf312b5ac55dbf3f68a4e12b785`.
  A wheel for the newer `a9b727a6` base was unavailable; the failed exact-base
  fetch is recorded in `artifacts/engram-shared-nvl72/runtime_install_final.log`.
- Model: `/data/models/DeepSeek-V4.1-Flash` with `config.json` SHA256
  `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`
  and index SHA256
  `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
  Two table tensors carry 768,022,850 rows total, width 256 FP8 bytes plus
  eight E8M0 scale bytes per row.
- Real P/D: P TP2 on GPUs 0-1, D TP2 on GPUs 2-3, DP1 each, NixlConnector,
  FP8 KV, `FLASHINFER_MLA_SPARSE_DSV41`, MXFP4 indexer, lazy safetensors,
  `max_model_len=32768`, `max_num_seqs=2`, 8192 batched tokens, 2 GiB KV/rank,
  eager, prefix cache disabled. The control used ordinary per-instance pinned
  host tables; the candidate used one persistent FABRIC owner. Request:
  14 prompt tokens and 32 completion tokens, temperature zero.

## Results by evidence level

| Level | Observation | Limit |
| --- | --- | --- |
| Component correctness | Separate-process FABRIC import, real checkpoint first/middle/last rows through the production Engram gather, and CUDA Graph replay passed for both layers. | Same host only. |
| Physical backing | Owner allocated 202,758,032,400 table bytes (188.833 GiB), 202,758,946,816 with VMM padding. Ordinary P and D each allocate the full table across TP ranks: 405,516,064,800 bytes total (377.666 GiB). Saving for two instances: 188.833 GiB of table backing. | Count is table backing, not total RSS, page cache, or HBM. |
| Local serving correctness | Two independent TP2 instances attached one generation and served concurrent requests. Real P-to-D NIXL handoff passed on the final PR head; candidate and ordinary control produced the same text and 32 completion tokens. P, D, and owner exited cleanly. | One P/D request per arm; not a performance result. |
| Model canary | First eight saved five-shot GSM8K questions: ordinary 8/8, shared 8/8, zero truncations, eight identical extracted answers. | Full token sequences matched 0/8; no bitwise or broad quality claim. |
| Test suite | 52 focused config/loader/owner tests, 137 Engram kernel/distributed tests, and applicable pre-commit hooks passed. | Upstream CI pre-run gate did not execute its pre-commit job. |

The paired P/D request took ordinary 0.756 s prefill / 8.869 s decode and
shared 0.798 s / 8.972 s on their respective sequential runs. These are
diagnostic single samples. They do not establish a speedup, regression bound,
or sustainable QPS. The earlier single-instance eager and decode-graph fixed
prompt controls matched token IDs exactly across storage modes.

A later bounded local P/D screen used one warmup and three retained requests
per prompt bucket, with ordinary first and shared second. Prefix caching was
disabled. Both arms used the same prompt hashes, actual token counts, 32
completion tokens, and NIXL route. All nine retained completion text hashes
matched. The table reports median HTTP phase elapsed seconds; it does not
isolate GPU kernel time.

| Prompt tokens | P ordinary | P shared | D ordinary | D shared | P shared/control | D shared/control |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 0.174 | 0.179 | 5.632 | 5.548 | 1.029 | 0.985 |
| 749 | 0.178 | 0.184 | 5.764 | 5.748 | 1.036 | 0.997 |
| 2,978 | 0.190 | 0.188 | 5.859 | 5.767 | 0.985 | 0.984 |

The preset screen was no more than 20% median regression in either P or D for
any bucket; all three buckets passed. This is one sequential, single-host
A/B with three retained samples per bucket. It supports no large local
regression under this low-concurrency workload, not a stable speedup, peak
QPS, tail SLO, or cross-host NVL72 performance result. Raw JSON SHA256:
ordinary `480d57fde7d160f6b2c6430ccb3983e628789fbe5ff4ff93d9a5164129398d91`;
shared `d1357c34ad8ce4c62a5c8b1c449df4b3e642551d8dbab2070c2678cb7a456f14`.

The initial one-NUMA owner allocation failed at 80.0/94.4 GiB on node 0 with
`CUDA_ERROR_OUT_OF_MEMORY` despite large reported `MemAvailable`; physical
`MemFree` was much smaller. The provider now places separate FABRIC chunks
across the two Grace nodes. A forced two-node small mapping and the complete
real-table publication then passed. This is an architecture/resource finding,
not evidence that every NVL72 host has sufficient backing capacity.

## Reproduce and artifacts

All scripts and raw outputs are under
`/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/`. In particular:

- `full_checkpoint_probe.py`, `full_checkpoint_probe_striped.log`: complete
  owner loading and direct real-row/graph validation.
- `smoke_ipc_lookup.py`, `smoke_ipc_lookup_exact.log` and `smoke_striped.py`,
  `smoke_striped.log`: separate-process direct gather and per-chunk NUMA mapping.
- `smoke_tp_layouts.py`, `smoke_tp_layouts.log`: seven TP1/TP2/TP4 rank
  processes using one synthetic generation, direct gather, graph replay, and
  clean detach. This checks layout mapping locally, not full-model mixed-TP P/D.
- `run_two_instances.py`, `run_two_instances.log`: simultaneous independent
  full-model TP2 instances against one owner.
- `run_pd_smoke.py`, `run_pd_smoke_final.log`, `run_pd_smoke_baseline.log`,
  `pd_smoke_shared.json`, `pd_smoke_baseline.json`: matched local P/D route and
  raw request context. The first P/D attempt returned empty text and was
  retained separately; the rerun used a known text-producing prompt and
  explicit drain shutdown.
- `run_pd_measure_baseline.log`, `run_pd_measure_shared.log`,
  `pd_measure_baseline.json`, `pd_measure_shared.json`: bounded three-bucket
  local serving screen and exact per-request prompt/output context.
- `gsm8k_canary.py`, `gsm8k_baseline.json`, `gsm8k_shared.json`,
  `run_gsm8k_shared.log`: fixed answer-level canary and generated tokens.
- `engram_regression_final.log`: 137-test current-source Engram suite.

For the source-level tests:

```bash
cd /workspace/vllm_dsv41/vllm-engram-shared-nvl72
.venv/bin/python -m pytest tests/test_config.py -k engram \
  tests/model_executor/model_loader/test_ep_weight_filter.py::test_lazy_mixed_shard_skips_external_table_before_get_tensor \
  tests/models/test_deepseek_v41_shared_engram.py -q
.venv/bin/python -m pytest tests/kernels/test_engram.py \
  tests/distributed/test_engram_dp_shard.py -q
```

Cross-host NVL72 import and serving, active-graph owner failure, full model
quality evaluation, sleep/reload, and production SLO acceptance remain open.
The upstream draft is for human review; its CI `pre-run-check` stopped because
the author has two merged PRs and none of the workflow's ready/verified labels.
The workflow expressly tells AI agents not to request such a label.

## Follow-up: outer wrapper text route

The full DeepSeek-V4.1 multimodal wrapper was exercised with image input
disabled (`language_model_only=False`, `limit_mm_per_prompt={"image":0}`). Its
shared-table and ordinary-table TP2 runs each generated 32 tokens for two
fixed text prompts; token IDs matched 2/2. The owner drained normally. This
validates wrapper loading and parameter accounting, not image processing.
Raw results are `results/vl_wrapper_shared.json` and
`results/vl_wrapper_baseline.json`, with logs and captured scripts beside them.

## Follow-up: 8K/128K/512K performance analysis

See [`performance/`](performance/) for the matched long-context P/D controls,
rank-paired CUDA trace analysis, negative Engram-kernel evidence, and the
bounded P-only scheduling-budget result. This is a single-host configuration
finding and does not qualify NVL72 cross-host performance.

## Follow-up: owner drain and lost control

PR follow-up `46706ee5` adds structured attach/detach and drain-wait records
with bounded client identity. A small real-GPU check signaled owner shutdown
while a consumer was attached: normal detach released the lease and owner
exited zero. In the lost-control case, the consumer first synchronized and
unmapped, then dropped its socket; the owner retained an uncertain lease and
waited until the test operator force-stopped it. The captured script is
`scripts/smoke_owner_drain.py.txt`; `logs/owner_drain_normal.log` and
`logs/owner_drain_lost.log` show the state transitions. This does not test
owner loss while a graph is actively replaying or remote-host failure.
