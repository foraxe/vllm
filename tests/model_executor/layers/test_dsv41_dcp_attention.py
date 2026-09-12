# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia import dcp


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_owner_filter_compacts_valid_suffix_without_changing_input(rank):
    ids = torch.tensor([[2, 4, -1, 9, 8, 3], [-1, -1, -1, -1, -1, -1]])
    before = ids.clone()
    local, counts = dcp.localize_topk(ids, rank, 3)
    expected = [x // 3 for x in ids[0].tolist() if x >= 0 and x % 3 == rank]
    assert local[0].tolist() == expected + [-1] * (6 - len(expected))
    assert local[1].tolist() == [-1] * 6
    assert counts.tolist() == [len(expected), 0]
    assert torch.equal(ids, before)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("ratio", [1, 2])
def test_prefill_offsets_keep_requests_planes_and_causal_bounds_separate(rank, ratio):
    ids = torch.tensor([[0, 1, 2, -1, 4, 5]] * 4, dtype=torch.int32)
    pos = torch.tensor([5, 6, 20, 21])
    req = torch.tensor([0, 0, 1, 1])
    lens = torch.tensor([7, 22])
    gather = torch.tensor([7, 8])
    result, counts = dcp.prefill_indices(
        ids,
        pos,
        req,
        lens,
        gather,
        rank=rank,
        world_size=2,
        ratio=ratio,
        window=4,
        request_stride=64,
        swa_offset=32,
    )
    assert result.shape == (4, 128)
    for row in range(4):
        r, p = int(req[row]), int(pos[row])
        expected = [
            r * 64 + c // 2
            for c in ids[row].tolist()
            if 0 <= c < (p + 1) // ratio and c % 2 == rank
        ]
        if rank == 0:
            expected += [
                r * 64 + 32 + t - int(lens[r] - gather[r])
                for t in range(max(0, p - 3), p + 1)
            ]
        actual = result[row][result[row] >= 0].tolist()
        assert actual == expected
        assert int(counts[row]) == len(expected)
        assert all(r * 64 <= x < (r + 1) * 64 for x in actual)


def test_query_exchange_excludes_local_padding(monkeypatch):
    class Group:
        def all_gather(self, tensor, dim):
            assert tensor.shape == (2, 2, 8) and dim == 1
            return torch.cat([tensor, tensor * 2], dim=1)

    monkeypatch.setattr(dcp, "get_dcp_group", lambda: Group())
    q = torch.full((2, 64, 8), 99.0)
    q[:, :2] = 1.0
    result = dcp.gather_query(q, 2)
    assert result.shape == (2, 64, 8)
    assert torch.all(result[:, :2] == 1)
    assert torch.all(result[:, 2:4] == 2)
    assert torch.count_nonzero(result[:, 4:]) == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_merge_normalizes_empty_native_lse_and_applies_sink_once(monkeypatch, rank):
    torch.manual_seed(41)
    scores = torch.randn(4, 5)
    values = torch.randn(5, 3)
    sink = torch.tensor([-torch.inf, torch.inf, 1.0, -1.0])
    parts, lses = [], []
    for owner in range(2):
        local_scores = scores[:, owner::2]
        part = local_scores.softmax(-1) @ values[owner::2]
        parts.append(torch.stack([part, torch.full_like(part, torch.nan)]))
        lses.append(
            torch.stack([local_scores.logsumexp(-1), torch.full((4,), torch.inf)])
        )
    masked_lses = [x.clone() for x in lses]
    for x in masked_lses:
        x[1] = -torch.inf
    total = torch.stack(masked_lses).logsumexp(0)
    weighted = [
        torch.stack(
            [parts[i][0] * (lses[i][0] - total[0]).exp()[:, None], torch.zeros(4, 3)]
        )
        for i in range(2)
    ]

    class Group:
        world_size = 2
        rank_in_group = rank

        def all_gather(self, tensor, dim):
            torch.testing.assert_close(tensor, masked_lses[rank])
            return torch.cat(masked_lses, dim=0)

        def all_reduce(self, tensor):
            torch.testing.assert_close(tensor, weighted[rank])
            return weighted[0] + weighted[1]

    monkeypatch.setattr(dcp, "get_dcp_group", lambda: Group())
    result = dcp.merge_partial_output(
        parts[rank],
        lses[rank],
        torch.tensor([3 - rank, 0]),
        2,
        sink[rank * 2 : rank * 2 + 2],
    )
    den = torch.logaddexp(scores.logsumexp(-1), sink)
    expected = (scores - den[:, None]).exp() @ values
    expected = torch.stack([expected[rank * 2 : rank * 2 + 2], torch.zeros(2, 3)])
    torch.testing.assert_close(result, expected)
