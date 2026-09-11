# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import json
from pathlib import Path


def state_probe(worker, action, output_prefix=None):
    import torch

    from vllm.v1.worker.ubatching import dbo_current_ubatch_id

    if action == "read":
        Path(worker._index_probe_path).write_text(
            json.dumps(worker._index_probe_stats, indent=2)
        )
        return worker._index_probe_stats
    if action in ("control", "enable"):
        parallel = worker.model_runner.parallel_config
        thresholds = (
            (2**30, 2**30) if action == "control" else worker._index_probe_thresholds
        )
        parallel.dbo_decode_token_threshold, parallel.dbo_prefill_token_threshold = (
            thresholds
        )
        stats = worker._index_probe_stats
        stats["seen"].clear()
        stats["topk_checks"] = stats["candidate_checks"] = 0
        worker._index_probe_snapshots.clear()
        worker._index_probe_path = f"{output_prefix}.rank-{stats['ep_rank']}.json"
        return {"phase": action, "thresholds": thresholds}
    model = worker.model_runner.get_model()
    if action == "share":
        changed = 0
        for module in model.modules():
            for attr in ("_topk_indices_buffer", "_candidate_blocks"):
                buffer = module.__dict__.get(attr)
                if isinstance(buffer, torch.Tensor) and buffer.ndim == 3:
                    setattr(module, attr, buffer[0])
                    changed += 1
        worker._index_probe_stats["shared_storage_mode"] = True
        return changed
    stats = {"rank": worker.rank, "seen": [], "topk_checks": 0, "candidate_checks": 0}
    snapshots = {}
    worker._index_probe_snapshots = snapshots
    parallel = worker.model_runner.parallel_config
    worker._index_probe_thresholds = (
        parallel.dbo_decode_token_threshold,
        parallel.dbo_prefill_token_threshold,
    )
    from vllm.distributed import get_ep_group

    stats["ep_rank"] = get_ep_group().rank_in_group
    worker._index_probe_path = f"{output_prefix}.rank-{stats['ep_rank']}.json"
    worker._index_probe_stats = stats
    layers = {}
    for name, module in model.named_modules():
        for layer in (2, 3, 20, 24):
            if name.endswith(f".layers.{layer}.attn"):
                layers[layer] = module
    assert len(layers) == 4, sorted(layers)
    stats["storage_shape"] = list(layers[2]._topk_indices_buffer.shape)

    def capture_topk(module, args, output):
        uid = dbo_current_ubatch_id()
        if uid not in stats["seen"]:
            stats["seen"].append(uid)
        n = args[0].numel()
        snapshots["topk", uid] = module.topk_indices_buffer[:n].clone()

    def check_topk(module, args):
        uid = dbo_current_ubatch_id()
        n = args[0].numel()
        if not torch.equal(snapshots["topk", uid], module.topk_indices_buffer[:n]):
            Path(worker._index_probe_path + ".collision.json").write_text(
                json.dumps(
                    {
                        "kind": "topk",
                        "uid": uid,
                        "ep_rank": stats["ep_rank"],
                        "rows": n,
                        "shared_storage_mode": stats.get("shared_storage_mode", False),
                        "mismatches": int(
                            torch.count_nonzero(
                                snapshots["topk", uid] != module.topk_indices_buffer[:n]
                            ).item()
                        ),
                    }
                )
            )
        assert torch.equal(snapshots["topk", uid], module.topk_indices_buffer[:n]), (
            f"topk overwritten on rank {worker.rank}, microbatch {uid}"
        )
        stats["topk_checks"] += 1

    def capture_candidates(module, args, output):
        uid = dbo_current_ubatch_id()
        n = args[0].numel()
        snapshots["candidates", uid] = module.indexer.indexer_op.candidate_blocks[
            :n
        ].clone()

    def check_candidates(module, args):
        uid = dbo_current_ubatch_id()
        n = args[0].numel()
        if not torch.equal(
            snapshots["candidates", uid], module.indexer.indexer_op.candidate_blocks[:n]
        ):
            Path(worker._index_probe_path + ".collision.json").write_text(
                json.dumps(
                    {
                        "kind": "candidates",
                        "uid": uid,
                        "ep_rank": stats["ep_rank"],
                        "rows": n,
                        "shared_storage_mode": stats.get("shared_storage_mode", False),
                    }
                )
            )
        assert torch.equal(
            snapshots["candidates", uid], module.indexer.indexer_op.candidate_blocks[:n]
        ), f"candidates overwritten on rank {worker.rank}, microbatch {uid}"
        stats["candidate_checks"] += 1

    layers[2].register_forward_hook(capture_topk)
    layers[3].register_forward_pre_hook(check_topk)
    layers[20].register_forward_hook(capture_candidates)
    layers[24].register_forward_pre_hook(check_candidates)
    return stats


class IndexStateProbeExtension:
    def index_state_probe(self, action, output_prefix=None):
        return state_probe(self, action, output_prefix)


async def main(args):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.tokenizers import get_tokenizer
    from vllm.v1.engine.async_llm import AsyncLLM

    model_path = "/data/models/DeepSeek-V4.1-Flash"
    engine_args = AsyncEngineArgs(
        model=model_path,
        tokenizer_mode="deepseek_v41",
        worker_extension_cls="dbo_model_eval.IndexStateProbeExtension",
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        enable_dbo=args.dbo,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=16,
        language_model_only=True,
        max_model_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1073741824,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        kernel_config={
            "enable_jit_warmup": False,
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
        },
        seed=0,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        installed = await engine.collective_rpc(
            "index_state_probe", args=("install", str(Path(args.output).resolve()))
        )
        print("PROBE_INSTALLED", installed, flush=True)
        tok = get_tokenizer(model_path, tokenizer_mode="deepseek_v41")
        questions = [
            ("What is 17 times 19? Return only the integer.", "323"),
            ("What is the capital of France? Return only the city name.", "Paris"),
            ("What is 2 plus 2? Return only the integer.", "4"),
            ("Write the next integer after 99. Return only the integer.", "100"),
        ] * 3
        tokenized = [
            tok.apply_chat_template(
                [{"role": "user", "content": "The sky is blue. " * 240 + "\n" + q}],
                tokenize=True,
                add_generation_prompt=True,
                thinking=False,
            )
            for q, _ in questions
        ]

        async def request(i, q, expected, phase="positive"):
            ids = tokenized[i]
            last = None
            async for last in engine.generate(
                {"prompt_token_ids": ids},
                SamplingParams(temperature=0, max_tokens=32),
                request_id=f"{phase}-{i}",
            ):
                pass
            assert last is not None
            out = last.outputs[0]
            return {
                "i": i,
                "expected": expected,
                "text": out.text,
                "token_ids": out.token_ids,
                "input_tokens": len(ids),
                "pass": out.text.strip() == expected,
            }

        control_rows = None
        if args.paired:
            prefix = str(Path(args.output).resolve()) + ".control"
            await engine.collective_rpc("index_state_probe", args=("control", prefix))
            control_rows = await asyncio.gather(
                *(request(i, q, a, "control") for i, (q, a) in enumerate(questions))
            )
            await engine.collective_rpc("index_state_probe", args=("read",))
            control_stats = [
                json.loads(Path(f"{prefix}.rank-{rank}.json").read_text())
                for rank in range(4)
            ]
            Path(args.output + ".control.json").write_text(
                json.dumps({"outputs": control_rows, "probe": control_stats}, indent=2)
            )
            assert all(r["pass"] for r in control_rows)
            assert all(s["seen"] == [0] for s in control_stats), control_stats
            await engine.collective_rpc(
                "index_state_probe", args=("enable", str(Path(args.output).resolve()))
            )
            print("PAIRED_CONTROL_PASSED", flush=True)
        rows = await asyncio.gather(
            *(request(i, q, a) for i, (q, a) in enumerate(questions))
        )
        stats = await engine.collective_rpc("index_state_probe", args=("read",))
        stats = [
            json.loads(Path(f"{args.output}.rank-{rank}.json").read_text())
            for rank in range(4)
        ]
        result = {"dbo": args.dbo, "outputs": rows, "probe": stats}
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        assert all(row["pass"] for row in rows)
        if control_rows is not None:
            assert [r["token_ids"] for r in rows] == [
                r["token_ids"] for r in control_rows
            ]
        assert all(s["topk_checks"] and s["candidate_checks"] for s in stats)
        if args.dbo:
            assert all(sorted(s["seen"]) == [0, 1] for s in stats), stats
        if args.negative_control:
            assert args.dbo
            print(
                "SHARED_STORAGE_NEGATIVE_CONTROL",
                await engine.collective_rpc("index_state_probe", args=("share",)),
                flush=True,
            )
            try:
                await asyncio.gather(
                    *(
                        request(i, q, a, "negative")
                        for i, (q, a) in enumerate(questions)
                    )
                )
            except Exception as error:
                collisions = list(
                    Path(args.output).parent.glob(
                        Path(args.output).name + ".rank-*.json.collision.json"
                    )
                )
                assert collisions, (
                    "Negative run failed without a recorded buffer collision"
                )
                data = {
                    "error": str(error),
                    "collisions": [json.loads(p.read_text()) for p in collisions],
                }
                Path(args.output + ".negative.json").write_text(
                    json.dumps(data, indent=2)
                )
                print("EXPECTED_BUFFER_COLLISION", data, flush=True)
            else:
                raise AssertionError(
                    "Shared-storage negative control did not expose the race"
                )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbo", action="store_true")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
