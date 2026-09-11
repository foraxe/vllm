# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the published and lean multicast restorers on identical inputs."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(rank, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    root = Path(os.environ["PCP_RESTORE_REFERENCE_ROOT"])
    old = load("old_restore", root / "vllm/v1/worker/gpu/pcp_hidden_restore.py")
    from vllm.v1.worker.gpu.pcp_hidden_restore import PCPMulticastHiddenStateRestorer

    bench = load(
        "restore_bench", root / "benchmarks/kernels/bench_pcp_hidden_restore.py"
    )
    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for rows in (1, 16, 64, 256, 1024, 4096):
            for skew in (False, True):
                args = dict(
                    group=dist.group.WORLD,
                    device=torch.device("cuda", rank),
                    max_num_tokens=rows,
                    hidden_size=6144,
                    dtype=dtype,
                )
                baseline = old.PCPMulticastHiddenStateRestorer(**args)
                candidate = PCPMulticastHiddenStateRestorer(**args)
                n = rows if skew else (rows + 3) // 4
                hidden = torch.randn(
                    (max(rows * 2, 8), 6144), device="cuda", dtype=dtype
                )
                indices = torch.arange(n, device="cuda", dtype=torch.int64) * 2
                global_rows = torch.arange(rows, device="cuda")
                restore = (
                    global_rows if skew else (global_rows % 4) * n + global_rows // 4
                )

                def a(
                    baseline=baseline,
                    hidden=hidden,
                    indices=indices,
                    restore=restore,
                    rows=rows,
                ):
                    return baseline.restore_selected(
                        hidden, indices, restore, num_selected_rows=rows
                    )

                def b(
                    candidate=candidate,
                    hidden=hidden,
                    indices=indices,
                    restore=restore,
                    rows=rows,
                ):
                    return candidate.restore_selected(
                        hidden, indices, restore, num_selected_rows=rows
                    )

                aa, bb = a(), b()
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
                # Alternating buffers must keep the preceding result alive.
                saved = bb.clone()
                hidden.add_(1)
                b()
                torch.testing.assert_close(bb, saved, rtol=0, atol=0)
                # Same allocations and rendezvous handle isolate the code change.
                lean_on_baseline = types.MethodType(
                    PCPMulticastHiddenStateRestorer.restore_selected, baseline
                )

                def same_buffer_lean(
                    lean_on_baseline=lean_on_baseline,
                    hidden=hidden,
                    indices=indices,
                    restore=restore,
                    rows=rows,
                ):
                    return lean_on_baseline(
                        hidden, indices, restore, num_selected_rows=rows
                    )

                stats = bench._measure_pair_us(
                    a,
                    same_buffer_lean,
                    cpu_group=dist.group.WORLD,
                    warmup=8,
                    iterations=40,
                )
                results.append(
                    dict(
                        rows=rows,
                        skew=skew,
                        dtype=str(dtype),
                        baseline_us=stats[0][0],
                        lean_us=stats[1][0],
                        baseline_p95_us=stats[0][1],
                        lean_p95_us=stats[1][1],
                    )
                )
                baseline.close()
                candidate.close()
                candidate.close()
                try:
                    b()
                    raise AssertionError("closed restorer accepted a call")
                except RuntimeError:
                    pass
    if rank == 0:
        Path(os.environ["PCP_RESTORE_OUTPUT"]).write_text(json.dumps(results, indent=2))
        print(json.dumps(results), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    from vllm.utils.network_utils import get_open_port

    mp.spawn(worker, args=(get_open_port(),), nprocs=4, join=True)
