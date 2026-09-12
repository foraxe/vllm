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


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="accelerator required")
def test_dummy_attention_profiles_dcp_temporary_memory(monkeypatch):
    """Automatic KV sizing must observe the FP32 DCP merge, not a zero-only stub."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.nvidia import flashmla

    class Group:
        world_size = 2
        rank_in_group = 0

        def all_gather(self, tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        def all_reduce(self, tensor):
            return tensor.clone()

    monkeypatch.setattr(dcp, "get_dcp_group", lambda: Group())
    monkeypatch.setattr(
        flashmla, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    monkeypatch.setattr(
        flashmla,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (
                torch.empty(1, 1, 512, device="cuda", dtype=torch.bfloat16),
            )
        ),
    )

    def native_stub(q, kv, indices, sm_scale, attn_sink, out):
        return (
            out,
            torch.zeros(q.shape[:2], device=q.device),
            torch.full(q.shape[:2], torch.inf, device=q.device),
        )

    monkeypatch.setattr(flashmla, "flash_mla_sparse_fwd", native_stub)
    q = torch.zeros(128, 64, 512, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(q)
    layer = SimpleNamespace(
        compress_ratio=1,
        max_model_len=32,
        window_size=128,
        max_num_batched_tokens=128,
        topk_indices_buffer=torch.empty(128, 512, dtype=torch.int32, device="cuda"),
        max_image_tokens=0,
        PREFILL_CHUNK_SIZE=4,
        dcp_world_size=2,
        n_local_heads=2,
        scale=512**-0.5,
        attn_sink=torch.zeros(64, device="cuda"),
    )
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_stats()["allocated_bytes.all.current"]
    flashmla.DeepseekV4FlashMLAAttention.forward_mqa(
        layer, q, q, torch.empty(0, device="cuda"), output
    )
    torch.accelerator.synchronize()
    peak = torch.accelerator.memory_stats()["allocated_bytes.all.peak"]
    # Two full FP32 buffers are a lower bound on live merge scratch. The old
    # profiling path allocated neither and left the KV budget too large.
    assert peak - before >= 2 * output.numel() * 4
    assert torch.count_nonzero(output) == 0
