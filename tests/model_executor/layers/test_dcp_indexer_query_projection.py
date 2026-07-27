# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import project_indexer_query
from vllm.platforms import current_platform


def test_project_indexer_query_reconstructs_32_heads_across_dcp4(
    monkeypatch,
) -> None:
    rows = 2
    world_size = 4
    local_heads = 8
    n_head = world_size * local_heads
    head_dim = 128
    rank = 2
    shards = tuple(
        torch.full(
            (rows, local_heads * head_dim),
            fill_value=dcp_rank,
            dtype=torch.bfloat16,
        )
        for dcp_rank in range(world_size)
    )

    class FakeDcpGroup:
        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == -1
            torch.testing.assert_close(tensor, shards[rank])
            return torch.cat(shards, dim=dim)

    def local_projection(
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert query.shape == (rows, 1)
        return shards[rank], None

    monkeypatch.setattr(deepseek_v2, "get_dcp_group", lambda: FakeDcpGroup())
    actual = project_indexer_query(
        local_projection,
        torch.empty(rows, 1),
        n_head,
        head_dim,
        indexer_q_sharded=True,
    )
    expected = torch.cat(shards, dim=-1).view(rows, n_head, head_dim)

    assert actual.shape == (rows, 32, head_dim)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
@torch.inference_mode()
def test_dcp_indexer_query_shards_reconstruct_replicated_projection(
    default_vllm_config,
    dist_init,
) -> None:
    """Guard the weight-loader ordering used by DCP indexer query sharding."""
    torch.accelerator.set_device_index(0)
    torch.set_default_device("cuda:0")
    torch.manual_seed(0)

    dcp_size = 4
    input_size = 256
    output_size = 512
    weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)
    query = torch.randn(8, input_size, dtype=torch.bfloat16)

    local_outputs = []
    expected_weights = weight.chunk(dcp_size, dim=0)
    for dcp_rank in range(dcp_size):
        projection = ColumnParallelLinear(
            input_size,
            output_size,
            bias=False,
            params_dtype=torch.bfloat16,
            tp_rank=dcp_rank,
            tp_size=dcp_size,
        )
        projection.weight.weight_loader(projection.weight, weight)
        assert torch.equal(projection.weight, expected_weights[dcp_rank])

        local_output, _ = projection(query)
        local_outputs.append(local_output)

    sharded_output = torch.cat(local_outputs, dim=-1)
    replicated_output = F.linear(query, weight)
    torch.testing.assert_close(sharded_output, replicated_output)
