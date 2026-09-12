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

    import vllm.v1.worker.gpu.pcp_manager as pcp
    from vllm.distributed import get_pcp_group

    runner = worker.model_runner
    manager = runner.pcp_manager
    if mode is None:
        return manager._restore_probe_stats
    if mode == "close":
        manager._hidden_state_restorer.close()
        return None
    if not hasattr(runner, "_lucas_comparison"):

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        root = os.environ["PCP_RESTORE_REFERENCE_ROOT"] + "/vllm/v1/worker/gpu/"
        transport = load(
            "vllm.v1.worker.gpu.pcp_hidden_restore", root + "pcp_hidden_restore.py"
        )
        old = load("_current_multicast_manager", root + "pcp_manager.py")
        old_prompt = load("_current_prompt_worker", root + "sample/prompt_logprob.py")
        manager._hidden_state_restorer = transport.PCPMulticastHiddenStateRestorer(
            group=get_pcp_group().cpu_group,
            device=manager.device,
            max_num_tokens=4,
            hidden_size=6144,
            dtype=torch.bfloat16,
        )
        assert manager._restore_buffers is not None
        packed, gathered, outputs = manager._restore_buffers
        manager._hidden_state_restorer._packed_input = packed
        manager._hidden_state_restorer._multicast_storage = gathered
        manager._hidden_state_restorer._ordered_outputs = outputs
        manager._hidden_restore_idx_cpu = None
        manager._hidden_states_are_replicated = False
        manager._sample_rows_are_identity = False
        runner._lucas_comparison = (
            old,
            type(manager),
            pcp.maybe_restore_pcp_for_sampling,
            old_prompt.PromptLogprobsWorker,
            type(runner.prompt_logprobs_worker),
        )
    old, new, original_helper, old_prompt, new_prompt = runner._lucas_comparison
    cls = old.PCPManager if mode == "multicast" else new
    for name in (
        "_build_batch_layout",
        "restore_sample_hidden_states",
        "restore_hidden_states",
    ):
        setattr(manager, name, types.MethodType(getattr(cls, name), manager))
    manager.restore_full_hidden_states = types.MethodType(
        old.PCPManager.restore_full_hidden_states, manager
    )
    prompt_cls = old_prompt if mode == "multicast" else new_prompt
    for name in ("needs_prompt_hidden_states", "compute_prompt_logprobs"):
        setattr(
            runner.prompt_logprobs_worker,
            name,
            types.MethodType(getattr(prompt_cls, name), runner.prompt_logprobs_worker),
        )
    stats = {
        "mode": mode,
        "decode_calls": 0,
        "prefill_calls": 0,
        "decode_cpu_s": 0.0,
        "prefill_cpu_s": 0.0,
    }
    manager._restore_probe_stats = stats

    def restore(self, hidden, **kwargs):
        full = kwargs.get(
            "needs_full_hidden_states", kwargs.get("needs_prompt_hidden_states")
        )
        key = (
            "needs_prompt_hidden_states"
            if mode == "multicast"
            else "needs_full_hidden_states"
        )
        start = time.perf_counter()
        out = cls.restore_for_sampling(self, hidden, **{key: full})
        label = "prefill" if self._global_batch.has_prefill else "decode"
        stats[label + "_calls"] += 1
        stats[label + "_cpu_s"] += time.perf_counter() - start
        return out

    manager.restore_for_sampling = types.MethodType(restore, manager)

    def helper(runner, hidden, batch):
        if mode == "inline":
            return original_helper(runner, hidden, batch)
        return old.maybe_restore_pcp_for_sampling(
            runner.pcp_manager,
            hidden,
            batch,
            needs_prompt_hidden_states=(
                runner.batch_sharder is not None
                or runner.speculator is not None
                or runner.prompt_logprobs_worker is None
                or runner.prompt_logprobs_worker.needs_prompt_hidden_states(
                    old.maybe_get_pcp_global_batch(runner.pcp_manager, batch),
                    runner.req_states.prompt_len.np,
                )
            ),
        )

    pcp.maybe_restore_pcp_for_sampling = helper
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
    schedule = [
        ("warmup", mode) for mode in ("multicast", "inline", "inline", "multicast")
    ]
    for pair in range(6):
        order = ("multicast", "inline") if pair % 2 == 0 else ("inline", "multicast")
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
