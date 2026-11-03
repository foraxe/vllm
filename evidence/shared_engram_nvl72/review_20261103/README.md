# Current-main VMM owner review, 2026-11-03

Final vLLM source: `192d3200ad347d08b980fb118d16be74f1bcb5b4`, merged with upstream main `ddbe9beb75567698ae5ecf29937e269f9a6e9b20`. Draft PR: <https://github.com/vllm-project/vllm/pull/64082> . Original validation and performance evidence remain in the parent snapshot.

## Source and test review

- Resolved mainline changes to `EngramConfig` (including Qwen PLE disk offload), small host-page lookup sorting, and model post-load hooks. Current main now finalizes model weights after `load_weights`; external parameter accounting and lazy safetensors filtering remain intact. The VL wrapper delegates checkpoint filtering to its language model without duplicating finalization.
- Partial CUDA VMM map cleanup now checks unmap/address-free results. If cleanup fails, the consumer closes control without a detach, so the owner retains the uncertain lease. Two tests cover simulated driver failure and the real owner protocol. The TP-layout test uses real packed torch tensors through the production slice helper instead of bypassing a module constructor.
- On the final head, 11 owner/TP tests plus one mixed-shard loader test passed. The 48 Engram config tests passed. All applicable pre-commit hooks on the 16 PR files passed. These tests are CPU and boundary tests, not a substitute for the serving run.

## Four-GPU local serving recheck

One original DeepSeek-V4.1-Flash checkpoint, a single 188.833 GiB HOST_NUMA owner, P TP2 on GB200 GPUs 0/1 and D TP2 on GPUs 2/3, DP1 each. NixlConnector, eager execution, lazy safetensors, FP8 KV, 8K batched-token budget, max two sequences, and a 32K context. The script and compressed logs are included here. GPUs 0-3 were leased exclusively and released after teardown.

The one retained routed request completed with 14 prompt tokens and 32 decode tokens. The owner logged eight attaches and eight protocol detaches, no uncertain lease; owner and both server top-level processes exited zero. This checks local integration at the final source head. It is not a matched A/B, concurrency result or performance claim. The API server logs contain `EngineDeadError` during SIGINT shutdown after the response and after resource teardown began; this run did not establish warning-free shutdown. No serving error occurred before the request completed.

Runtime: source imported from this worktree, Torch 2.14.0+cu130; CUDA native binaries are inherited from the precompiled vLLM wheel at `40fe396b` as in the parent evidence, not rebuilt from this main commit. No native extension is changed in this PR. Model config SHA256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`; checkpoint index SHA256 `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.

Cross-host NVL72/IMEX import remains untested because only one host is available. The prior 8K, 128K, 512K, 1M and C128 data use older source and are not remeasured here.
