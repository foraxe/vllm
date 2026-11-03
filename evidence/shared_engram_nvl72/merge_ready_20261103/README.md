# Current-main shared Engram merge-readiness gates, 2026-11-03

vLLM source `b4f4a8120887ca773239e8d31ca618a00a67546d` includes upstream main `2173d7121fc6e8d756c884f46691546c61b60c57`. SGLang source `b0681ed0f354c15e2cae999f30040be2fdc8a698` includes upstream main `40a64e1b5ba00c73b7f8a123a903a2123b7d467c`. Model: original `/data/models/DeepSeek-V4.1-Flash`, config SHA256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`, index SHA256 `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`. One four-GB200 host; each GPU run held an exclusive lease on GPUs 0-3 and released it after teardown. vLLM used Torch 2.14.0+cu130 and SGLang used Torch 2.14.1+cu130 in separate runtime overlays. No exact-main native wheel was built for vLLM; its precompiled native libraries came from the older wheel recorded in the parent snapshot.

## Model-answer gate (vLLM)

Matched ordinary host-table and external-owner TP2 runs used the same saved first 32 five-shot GSM8K questions, greedy sampling, 512-token cap, eager execution, 32K max context, FP8 KV, and the same FlashInfer MLA sparse/indexer settings. Both modes scored 31/32 with zero truncations; all 32 extracted predictions matched. Full token streams matched 8/32, so the evidence is answer-level, not bitwise equivalence. The owner logged four attaches and four protocol detaches, zero uncertain leases, and exited zero. `compact_results.json` contains per-case predictions and output hashes without full generated text. Raw local reports remain under `/workspace/vllm_dsv41/artifacts/engram-shared-nvl72/merge_ready_eval_20261103/`.

## Serving gate (SGLang with vLLM owner)

Matched ordinary CPU and external-owner P TP2+D TP2 native-NIXL runs each completed one warmup and three retained 13-input/32-output requests, eager, max two running requests, 8K chunked prefill, 32K context. The three retained text and token-ID hashes matched across modes (3/3). One sequential arm order gave median P-to-D wall 4.461 s ordinary and 4.361 s shared; this is diagnostic, not a stable speedup claim. The shared owner logged eight attaches and eight protocol detaches, zero uncertain leases, and exited zero. P/D top-level processes exited zero in both arms. SGLang source imported from its worktree via PYTHONPATH; the provider came from the updated vLLM worktree. `compact_results.json` and compressed logs hold exact provenance and route evidence.

## Compatibility and remaining gates

Current vLLM main added direct weight reload. A new guard rejects direct reload, in-place reload and weight transfer when external Engram is configured, because the owner generation is immutable. The focused direct-reload test failed on the prior branch with `DID NOT RAISE` and passed after the guard. The 12 owner/loader tests, 48 Engram config tests, 13 direct-reload compatibility tests, six SGLang external unit tests, two history tests and all applicable PR-file pre-commit hooks passed on these heads.

A second GB200 host in the same IMEX/NVLink domain is still required to validate cross-host FABRIC import, remote Grace read latency, failure handling and multi-host serving. Upstream CI and human line review remain separate gates. This local evidence does not establish production SLOs or stable speedup.
