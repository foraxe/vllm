# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded full-model engine benchmark for PCP final-row restoration."""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm import LLM, SamplingParams


def set_restore_mode(worker, mode):
    import importlib.util
    import os
    import sys
    import types

    from vllm.distributed import get_pcp_group

    runner = worker.model_runner
    if mode is None:
        return runner.pcp_manager._restore_probe_stats
    if mode == "close":
        runner._restore_comparison[0]._hidden_state_restorer.close()
        return None
    if not hasattr(runner, "_restore_comparison"):
        candidate = runner.pcp_manager

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        root = os.environ["PCP_RESTORE_REFERENCE_ROOT"] + "/vllm/v1/worker/gpu/"
        transport = load("_published_restore", root + "pcp_hidden_restore.py")
        module = load("_published_manager", root + "pcp_manager.py")
        module.PCPMulticastHiddenStateRestorer = (
            transport.PCPMulticastHiddenStateRestorer
        )
        restorer = transport.PCPMulticastHiddenStateRestorer(
            group=get_pcp_group().cpu_group,
            device=candidate.device,
            max_num_tokens=4,
            hidden_size=6144,
            dtype=torch.bfloat16,
        )
        baseline = module.PCPManager(
            pcp_world_size=candidate.pcp_world_size,
            pcp_rank=candidate.pcp_rank,
            device=candidate.device,
            req_states=candidate._req_states,
            max_num_reqs=4,
            max_num_tokens=4096,
            block_tables=candidate._block_tables,
            hidden_state_restorer=restorer,
        )
        runner._lean_manager_class = type(candidate)
        runner._published_manager_class = type(baseline)
        runner._lean_transport_class = type(candidate._hidden_state_restorer)
        runner._published_transport_class = type(restorer)
        runner._restore_comparison = baseline, candidate
    # Share input buffers, metadata storage and multicast storage. Change only
    # method implementations, not the model's physical input allocation.
    manager = runner._restore_comparison[0]
    manager_class = (
        runner._lean_manager_class
        if mode == "lean"
        else runner._published_manager_class
    )
    for name in (
        "_build_batch_layout",
        "restore_sample_hidden_states",
        "restore_full_hidden_states",
        "restore_hidden_states",
    ):
        setattr(manager, name, types.MethodType(getattr(manager_class, name), manager))
    manager._comparison_original_restore = types.MethodType(
        manager_class.restore_for_sampling, manager
    )
    transport_class = (
        runner._lean_transport_class
        if mode == "lean"
        else runner._published_transport_class
    )
    manager._hidden_state_restorer.restore_selected = types.MethodType(
        transport_class.restore_selected, manager._hidden_state_restorer
    )
    runner.pcp_manager = manager
    stats = {
        "mode": mode,
        "decode_calls": 0,
        "prefill_calls": 0,
        "decode_cpu_s": 0.0,
        "prefill_cpu_s": 0.0,
    }
    manager._restore_probe_stats = stats

    def restore(self, hidden, *, needs_prompt_hidden_states):
        start = time.perf_counter()
        result = self._comparison_original_restore(
            hidden, needs_prompt_hidden_states=needs_prompt_hidden_states
        )
        label = "prefill" if self._global_batch.has_prefill else "decode"
        stats[label + "_calls"] += 1
        stats[label + "_cpu_s"] += time.perf_counter() - start
        return result

    manager.restore_for_sampling = types.MethodType(restore, manager)
    return {"mode": mode, "rank": manager.pcp_rank}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--synchronized-batch", action="store_true")
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
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        num_gpu_blocks_override=1024,
        enforce_eager=True,
        disable_log_stats=False,
        disable_custom_all_reduce=True,
        moe_backend="flashinfer_cutlass",
        attention_config={"mla_prefill_backend": "FLASH_ATTN"},
        kernel_config={"enable_jit_warmup": False, "enable_flashinfer_autotune": False},
        compilation_config={"mode": "NONE", "cudagraph_mode": "NONE"},
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    prompts = []
    for index in range(4):
        text = (
            f"Section {index}. Explain context parallel inference "
            "and its memory requirements. " * 200
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)[:1024]
        assert len(tokens) == 1024
        prompts.append({"prompt_token_ids": tokens})
    sampling = SamplingParams(
        temperature=0, max_tokens=32, min_tokens=32, ignore_eos=True
    )

    trials = []
    schedule = [("warmup", mode) for mode in ("published", "lean", "lean", "published")]
    for pair in range(6):
        order = ("published", "lean") if pair % 2 == 0 else ("lean", "published")
        schedule.extend((pair, mode) for mode in order)
    for pair, mode in schedule:
        llm.collective_rpc(set_restore_mode, args=(mode,))
        if args.synchronized_batch:
            llm.sleep(level=0, mode="keep")
        start = time.perf_counter()
        if args.synchronized_batch:
            llm.enqueue(prompts, sampling, use_tqdm=False)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(use_tqdm=False)
        else:
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        stats = llm.collective_rpc(set_restore_mode, args=(None,))
        row = {
            "pair": pair,
            "mode": mode,
            "elapsed_s": elapsed,
            "ttft_ms": [o.metrics.first_token_latency * 1000 for o in outputs],
            "tpot_ms": [
                (o.metrics.last_token_ts - o.metrics.first_token_ts) * 1000 / 31
                for o in outputs
            ],
            "output_tokens_per_s": 128 / elapsed,
            "tokens": [list(o.outputs[0].token_ids) for o in outputs],
            "restore_stats": stats,
        }
        if pair != "warmup":
            trials.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("tokens", "restore_stats")}
            ),
            flush=True,
        )
        args.output.write_text(json.dumps({"trials": trials}, indent=2) + "\n")

    for pair in range(6):
        paired = [r for r in trials if r["pair"] == pair]
        assert paired[0]["tokens"] == paired[1]["tokens"]
        assert all(
            s["prefill_calls"] == 1 and s["decode_calls"] == 31
            for r in paired
            for s in r["restore_stats"]
        )
    llm.collective_rpc(set_restore_mode, args=("close",))


if __name__ == "__main__":
    main()
