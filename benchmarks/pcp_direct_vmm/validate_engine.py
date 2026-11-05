# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded full-model direct-final VMM graph smoke; no performance claim."""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def inspect(worker):
    runner = worker.model_runner
    domain = runner.pcp_vmm_domain
    cache = runner.extensible_kv_cache
    assert domain is not None and domain.owner is cache.buffer
    assert runner.pcp_manager.use_local_kv_slot_mappings
    assert runner.cudagraph_manager.graphs
    if not hasattr(runner, "_direct_smoke_replays"):
        runner._direct_smoke_replays = 0
        replay = runner.cudagraph_manager.run_fullgraph

        def counted_replay(*args, **kwargs):
            runner._direct_smoke_replays += 1
            return replay(*args, **kwargs)

        runner.cudagraph_manager.run_fullgraph = counted_replay
    layers = runner.vllm_config.compilation_config.static_forward_context.values()
    assert all(
        layer.pcp_vmm_domain is domain
        for layer in layers
        if getattr(layer, "use_pcp", False)
    )
    return {
        "rank": domain.group.rank_in_group,
        "committed_blocks": cache.num_committed_blocks,
        "capacity_blocks": cache.capacity_blocks,
        "owner_va": cache.buffer.base_ptr,
        "peer_table": domain.peers.pointers.data_ptr(),
        "epoch": domain.epoch.item(),
        "graph_replays": runner._direct_smoke_replays,
        "graph_mode": str(runner.vllm_config.compilation_config.cudagraph_mode),
    }


def use_collective_oracle(worker):
    # Test-only: compare native KV collectives in eager mode on the same weights.
    runner = worker.model_runner
    for layer in runner.vllm_config.compilation_config.static_forward_context.values():
        if getattr(layer, "use_pcp", False):
            layer.pcp_vmm_domain = None
    runner.pcp_manager.use_local_kv_slot_mappings = False
    runner.cudagraph_manager._graphs_captured = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("pcp_direct_vmm_smoke.json")
    )
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        load_format="fastsafetensors",
        tensor_parallel_size=1,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        kv_cache_dtype="fp8",
        enable_extensible_kv_cache=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        moe_backend="flashinfer_cutlass",
        attention_config={"mla_prefill_backend": "FLASH_ATTN"},
        kernel_config={"enable_jit_warmup": False, "enable_flashinfer_autotune": False},
        compilation_config={
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        },
        seed=0,
    )
    tokens = llm.get_tokenizer().encode("Explain context parallel inference. " * 30)[
        :64
    ]
    params = SamplingParams(temperature=0, min_tokens=4, max_tokens=4, ignore_eos=True)
    results = []
    for count in (1, 2, 4, 1):
        before = llm.collective_rpc(inspect)
        out = llm.generate(
            [{"prompt_token_ids": tokens}] * count, params, use_tqdm=False
        )
        after = llm.collective_rpc(inspect)
        assert all(
            a["epoch"] > b["epoch"]
            and a["owner_va"] == b["owner_va"]
            and a["peer_table"] == b["peer_table"]
            for a, b in zip(after, before)
        )
        assert all(
            a["graph_replays"] > b["graph_replays"] for a, b in zip(after, before)
        )
        results.append(
            {
                "batch": count,
                "before": before,
                "after": after,
                "tokens": [list(o.outputs[0].token_ids) for o in out],
            }
        )
    assert results[0]["tokens"] == results[-1]["tokens"]
    llm.collective_rpc(use_collective_oracle)
    oracle = llm.generate([{"prompt_token_ids": tokens}], params, use_tqdm=False)
    oracle_tokens = [list(o.outputs[0].token_ids) for o in oracle]
    assert results[0]["tokens"] == oracle_tokens
    results.append({"collective_eager_oracle_tokens": oracle_tokens})
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("EXTENSIBLE_DIRECT_ENGINE_GRAPH_SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
