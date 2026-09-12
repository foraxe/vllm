# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Published multicast versus Lucas's production compact PCP AllGather."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, port):
    torch.accelerator.set_device_index(rank)
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_pcp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(4, rank, f"tcp://127.0.0.1:{port}", rank)
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(
            tensor_model_parallel_size=1, prefill_context_model_parallel_size=4
        )
    group = get_pcp_group()
    root = Path(os.environ["PCP_RESTORE_WORKSPACE"])
    old = load(
        "vllm.v1.worker.gpu.pcp_hidden_restore",
        root / "vllm-pcp-restore-lean/vllm/v1/worker/gpu/pcp_hidden_restore.py",
    )
    old_manager_module = load(
        "current_pcp_manager",
        root / "vllm-pcp-restore-lean/vllm/v1/worker/gpu/pcp_manager.py",
    )
    bench = load(
        "bench",
        root
        / "vllm-pcp-multicast-current/benchmarks/kernels/bench_pcp_hidden_restore.py",
    )
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (1, 16, 64, 256, 1024, 4096):
            for skew in (False, True):
                restorer = old.PCPMulticastHiddenStateRestorer(
                    group=group.cpu_group,
                    device=torch.device("cuda", rank),
                    max_num_tokens=rows,
                    hidden_size=6144,
                    dtype=dtype,
                )
                n = rows if skew else (rows + 3) // 4
                hidden = torch.randn(
                    (max(rows * 2, 8), 6144), device="cuda", dtype=dtype
                )
                index = torch.arange(n, device="cuda") * 2
                r = torch.arange(rows, device="cuda")
                order = r if skew else (r % 4) * n + r // 4
                manager = PCPManager(4, rank, torch.device("cuda", rank))
                manager._global_batch = SimpleNamespace(has_prefill=True)
                manager._sample_local_row_idx = index
                manager._sample_restore_idx = order
                manager._restore_buffers = (
                    restorer._packed_input,
                    restorer._multicast_storage,
                    restorer._ordered_outputs,
                )
                manager._restore_group_name = group.cpu_group.group_name
                old_manager = old_manager_module.PCPManager(
                    4, rank, torch.device("cuda", rank), hidden_state_restorer=restorer
                )
                old_manager._global_batch = manager._global_batch
                old_manager._sample_local_row_idx = index
                old_manager._sample_restore_idx = order

                def baseline(old_manager=old_manager, hidden=hidden):
                    return old_manager.restore_sample_hidden_states(hidden)

                def candidate(manager=manager, hidden=hidden):
                    return manager.restore_sample_hidden_states(hidden)

                expected = baseline().clone()
                torch.testing.assert_close(expected, candidate(), rtol=0, atol=0)
                a, b = bench._measure_pair_us(
                    baseline,
                    candidate,
                    cpu_group=group.cpu_group,
                    warmup=8,
                    iterations=30,
                )
                results.append(
                    dict(
                        dtype=str(dtype),
                        rows=rows,
                        skew=skew,
                        multicast_us=a[0],
                        collective_us=b[0],
                        multicast_p95=a[1],
                        collective_p95=b[1],
                    )
                )
                restorer.close()
    if rank == 0:
        Path(os.environ["PCP_RESTORE_OUTPUT"]).write_text(json.dumps(results, indent=2))
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from vllm.utils.network_utils import get_open_port

    mp.spawn(worker, args=(get_open_port(),), nprocs=4, join=True)
