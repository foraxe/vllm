# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.utils import get_open_port


def _all_gather_cat(value: torch.Tensor, dim: int) -> torch.Tensor:
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    return torch.cat(gathered, dim=dim)


def _candidate_worker(rank: int, world: int, port: int) -> None:
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
        select_dcp_candidate_blocks,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        device = torch.device("cuda", rank)
        allocated = [3, 19, 25, 43]
        visible_values = [0, 2, 17, 41]
        visible = torch.tensor(visible_values, dtype=torch.int32, device=device)
        local_allocated = [(length + world - 1 - rank) // world for length in allocated]
        starts_values = [0]
        for length in local_allocated[:-1]:
            starts_values.append(starts_values[-1] + length)
        starts = torch.tensor(
            starts_values,
            dtype=torch.int32,
            device=device,
        )
        local_visible_values = [
            (length + world - 1 - rank) // world for length in visible_values
        ]
        local_visible = torch.tensor(
            local_visible_values,
            dtype=torch.int32,
            device=device,
        )
        ends = starts + local_visible
        rows = len(allocated)
        local_logits = torch.full(
            (rows, sum(local_allocated)), -torch.inf, device=device
        )
        dense_logits = torch.full((rows, max(allocated)), -torch.inf, device=device)
        topk_blocks = 4
        block_size = 8

        for replay in range(2):
            for row, global_len in enumerate(visible_values):
                global_ids = torch.arange(global_len, device=device)
                values = (global_ids * 17 + row * 13 + replay).float()
                if replay and row == 2:
                    values[5] = float("nan")
                dense_logits[row].fill_(-torch.inf)
                dense_logits[row, :global_len] = values
                owned = global_ids % world == rank
                local_logits[row].fill_(-torch.inf)
                start = starts_values[row]
                local_logits[row, start : start + local_visible_values[row]] = values[
                    owned
                ]

            actual = torch.empty((rows, topk_blocks), dtype=torch.int32, device=device)
            select_dcp_candidate_blocks(
                local_logits,
                starts,
                ends,
                visible,
                topk_blocks,
                block_size,
                actual,
                rank,
                world,
                _all_gather_cat,
                chunk_blocks=2,
            )
            expected = torch.empty_like(actual)
            select_candidate_blocks(
                dense_logits,
                None,
                visible,
                topk_blocks,
                block_size,
                expected,
            )
            torch.testing.assert_close(
                actual.sort(dim=1).values,
                expected.sort(dim=1).values,
                rtol=0,
                atol=0,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.accelerator.is_available() or torch.accelerator.device_count() < 4,
    reason="four CUDA devices required",
)
@pytest.mark.parametrize("world", [2, 4])
def test_dsv41_dcp_candidate_blocks_match_dense_reference(world: int) -> None:
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
    mp.spawn(_candidate_worker, args=(world, get_open_port()), nprocs=world, join=True)
