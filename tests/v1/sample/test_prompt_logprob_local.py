# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for GLM-only TP-local target prompt logprobs."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.multiprocessing

from vllm.model_executor.layers.logits_processor import LocalLogits
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from vllm.v1.worker.gpu.model_runner import _get_prompt_logprobs_local_logits_fn
from vllm.v1.worker.gpu.sample import prompt_logprob


class _SingleRankTPGroup:
    world_size = 1
    device_group = None

    @staticmethod
    def all_reduce(values: torch.Tensor) -> torch.Tensor:
        return values

    @staticmethod
    def all_gather(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        del dim
        return values


class _GlooTPGroup:
    world_size = 2
    device_group = torch.distributed.group.WORLD

    @staticmethod
    def all_reduce(values: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(values, group=torch.distributed.group.WORLD)
        return values

    @staticmethod
    def all_gather(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
        gathered = [torch.empty_like(values) for _ in range(2)]
        torch.distributed.all_gather(
            gathered, values, group=torch.distributed.group.WORLD
        )
        return torch.cat(gathered, dim=dim)


def _local_logits(
    logits: torch.Tensor, start: int = 0, org_width: int | None = None
) -> LocalLogits:
    org_width = logits.shape[-1] if org_width is None else org_width
    return LocalLogits(
        logits=logits,
        shard_indices=VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=start + logits.shape[-1],
            padded_added_vocab_start_index=start + logits.shape[-1],
            padded_added_vocab_end_index=start + logits.shape[-1],
            org_vocab_start_index=start,
            org_vocab_end_index=start + org_width,
            added_vocab_start_index=start + org_width,
            added_vocab_end_index=start + org_width,
        ),
    )


def test_target_only_matches_one_rank_log_softmax(monkeypatch):
    monkeypatch.setattr(prompt_logprob, "get_tp_group", _SingleRankTPGroup)
    logits = torch.tensor([[1.0, -2.0, 3.0], [0.5, 2.0, -1.0]])
    target_ids = torch.tensor([2, 0])

    result = prompt_logprob.compute_distributed_token_logprobs(
        _local_logits(logits), target_ids
    )

    expected = torch.log_softmax(logits, -1)[torch.arange(2), target_ids]
    torch.testing.assert_close(result.logprobs[:, 0], expected)
    assert result.logprob_token_ids.tolist() == [[2], [0]]
    assert result.selected_token_ranks.tolist() == [1, 2]
    assert result.selected_token_ranks.dtype == torch.int64


def test_target_only_excludes_dominant_padded_and_added_tail(monkeypatch):
    monkeypatch.setattr(prompt_logprob, "get_tp_group", _SingleRankTPGroup)
    local = LocalLogits(
        logits=torch.tensor([[1.0, 2.0, -float("inf"), -float("inf"), 1e6]]),
        shard_indices=VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=4,
            padded_org_vocab_end_index=8,
            padded_added_vocab_start_index=6,
            padded_added_vocab_end_index=7,
            org_vocab_start_index=4,
            org_vocab_end_index=6,
            added_vocab_start_index=6,
            added_vocab_end_index=7,
        ),
    )

    result = prompt_logprob.compute_distributed_token_logprobs(local, torch.tensor([5]))

    torch.testing.assert_close(
        result.logprobs[:, 0], torch.log_softmax(torch.tensor([[1.0, 2.0]]), -1)[:, 1]
    )
    assert result.selected_token_ranks.tolist() == [1]


def _chunking_fallback_test(monkeypatch, num_prompt_logprobs: int) -> None:
    calls = SimpleNamespace(gathered=0, local=0)
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return logits[: hidden_states.shape[0]]

    def local(_: torch.Tensor) -> LocalLogits:
        calls.local += 1
        raise AssertionError(
            "gathered prompt-logprob requests must not project locally"
        )

    def fake_topk(
        prompt_logits: torch.Tensor,
        requested_num: int,
        _: torch.Tensor,
        **__: object,
    ) -> prompt_logprob.LogprobsTensors:
        assert requested_num == (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                prompt_logits.shape[0], requested_num + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(prompt_logits.shape[0], requested_num + 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.tensor([1, 0]),
        torch.empty(2, 1),
        gathered,
        local,
        num_prompt_logprobs,
    )
    assert calls.gathered == 1
    assert calls.local == 0


@pytest.mark.parametrize("num_prompt_logprobs", [1, 5])
def test_chunking_gathers_for_short_prompt_topn(monkeypatch, num_prompt_logprobs):
    _chunking_fallback_test(monkeypatch, num_prompt_logprobs=num_prompt_logprobs)


def test_chunking_gathers_for_full_prompt_logprobs_minus_one(monkeypatch):
    """``prompt_logprobs=-1`` never invokes the local target-only route."""
    _chunking_fallback_test(monkeypatch, num_prompt_logprobs=-1)


def test_chunking_gathers_full_minus_one_at_or_above_local_threshold(monkeypatch):
    """The finite-N local eligibility threshold never changes ``-1`` routing."""
    rows = prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS
    calls = SimpleNamespace(gathered=0, local=0)

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return torch.zeros(hidden_states.shape[0], 2)

    def local(_: torch.Tensor) -> LocalLogits:
        calls.local += 1
        raise AssertionError("prompt_logprobs=-1 must not project local logits")

    def forbidden_distributed(*_: object) -> prompt_logprob.LogprobsTensors:
        raise AssertionError("prompt_logprobs=-1 must not use compact Top-N")

    def gathered_topk(
        prompt_logits: torch.Tensor,
        requested_num: int,
        token_ids: torch.Tensor,
        **_: object,
    ) -> prompt_logprob.LogprobsTensors:
        assert requested_num == prompt_logits.shape[-1]
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                token_ids.shape[0], requested_num + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(token_ids.shape[0], requested_num + 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_distributed_topk_scores", forbidden_distributed
    )
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", gathered_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs=-1,
    )

    assert calls.gathered == 1
    assert calls.local == 0


@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_uses_local_target_only_at_minimum_row_count(
    monkeypatch, logprobs_mode
):
    calls = SimpleNamespace(gathered=0, local=0, distributed=0)

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return torch.zeros(hidden_states.shape[0], 2)

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local += 1
        return _local_logits(torch.zeros(hidden_states.shape[0], 2))

    def distributed(
        _: LocalLogits, token_ids: torch.Tensor, mode: object
    ) -> prompt_logprob.LogprobsTensors:
        assert mode == logprobs_mode
        calls.distributed += 1
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(token_ids.shape[0], 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_distributed_token_logprobs", distributed
    )
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS, dtype=torch.int64),
        torch.empty(prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
        logprobs_mode=logprobs_mode,
    )

    assert calls.gathered == 0
    assert calls.local == calls.distributed == 1


@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_gathers_below_minimum_without_local_projection(
    monkeypatch, logprobs_mode
):
    calls = SimpleNamespace(gathered=0, local=0)
    logged: list[str] = []
    rows = prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS - 1

    class Logger:
        @staticmethod
        def info_once(message: str, *args: object) -> None:
            logged.append(message % args)

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        return torch.zeros(hidden_states.shape[0], 2)

    def local(_: torch.Tensor) -> LocalLogits:
        calls.local += 1
        raise AssertionError("short chunks must not project local logits")

    def fake_topk(
        prompt_logits: torch.Tensor, _: int, token_ids: torch.Tensor, **kwargs: object
    ) -> prompt_logprob.LogprobsTensors:
        assert kwargs["logits_mode"] == (logprobs_mode == "raw_logits")
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(prompt_logits.shape[0], 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "logger", Logger())
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
        logprobs_mode=logprobs_mode,
    )

    assert calls.gathered == 1
    assert calls.local == 0
    assert logged == [
        (
            f"{prompt_logprob.SHORT_CHUNK_GATHERED_MARKER} rows={rows} "
            f"threshold={prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS}"
        )
    ]


def test_chunking_mixed_1025_rows_uses_local_prefix_and_gathered_tail(monkeypatch):
    calls = SimpleNamespace(gathered_rows=[], local_rows=[], distributed_rows=[])
    rows = 1025

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered_rows.append(hidden_states.shape[0])
        return torch.zeros(hidden_states.shape[0], 2)

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local_rows.append(hidden_states.shape[0])
        return _local_logits(torch.zeros(hidden_states.shape[0], 2))

    def distributed(
        _: LocalLogits, token_ids: torch.Tensor, __: object
    ) -> prompt_logprob.LogprobsTensors:
        calls.distributed_rows.append(token_ids.shape[0])
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(token_ids.shape[0], 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    def fake_topk(
        prompt_logits: torch.Tensor, _: int, token_ids: torch.Tensor, **__: object
    ) -> prompt_logprob.LogprobsTensors:
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=token_ids[:, None],
            logprobs=torch.zeros(prompt_logits.shape[0], 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(
        prompt_logprob, "compute_distributed_token_logprobs", distributed
    )
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs=0,
    )

    assert calls.local_rows == calls.distributed_rows == [1024]
    assert calls.gathered_rows == [1]


@pytest.mark.parametrize("num_prompt_logprobs", [1, 5])
@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_uses_local_finite_topn_at_minimum_row_count(
    monkeypatch, num_prompt_logprobs, logprobs_mode
):
    calls = SimpleNamespace(gathered=0, local=0, distributed=0)
    rows = prompt_logprob.MIN_LOCAL_PROMPT_LOGPROB_ROWS

    def gathered(_: torch.Tensor) -> torch.Tensor:
        calls.gathered += 1
        raise AssertionError("eligible finite Top-N chunks must remain TP-local")

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local += 1
        return _local_logits(torch.zeros(hidden_states.shape[0], 6))

    def distributed(
        _: LocalLogits, num_logprobs: int, token_ids: torch.Tensor, mode: object
    ) -> prompt_logprob.LogprobsTensors:
        assert num_logprobs == num_prompt_logprobs
        assert mode == logprobs_mode
        calls.distributed += 1
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                token_ids.shape[0], num_logprobs + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(token_ids.shape[0], num_logprobs + 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_distributed_topk_scores", distributed)
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs,
        logprobs_mode,
    )

    assert calls.gathered == 0
    assert calls.local == calls.distributed == 1


@pytest.mark.parametrize("num_prompt_logprobs", [1, 5])
@pytest.mark.parametrize("logprobs_mode", ["raw_logits", "raw_logprobs"])
def test_chunking_finite_topn_gathers_short_tail(
    monkeypatch, num_prompt_logprobs, logprobs_mode
):
    calls = SimpleNamespace(gathered_rows=[], local_rows=[], distributed_rows=[])
    rows = 1025

    def gathered(hidden_states: torch.Tensor) -> torch.Tensor:
        calls.gathered_rows.append(hidden_states.shape[0])
        return torch.zeros(hidden_states.shape[0], 6)

    def local(hidden_states: torch.Tensor) -> LocalLogits:
        calls.local_rows.append(hidden_states.shape[0])
        return _local_logits(torch.zeros(hidden_states.shape[0], 6))

    def distributed(
        _: LocalLogits, num_logprobs: int, token_ids: torch.Tensor, mode: object
    ) -> prompt_logprob.LogprobsTensors:
        assert mode == logprobs_mode
        calls.distributed_rows.append(token_ids.shape[0])
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                token_ids.shape[0], num_logprobs + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(token_ids.shape[0], num_logprobs + 1),
            selected_token_ranks=torch.ones(token_ids.shape[0], dtype=torch.int64),
        )

    def fake_topk(
        prompt_logits: torch.Tensor,
        num_logprobs: int,
        _: torch.Tensor,
        **kwargs: object,
    ) -> prompt_logprob.LogprobsTensors:
        assert kwargs["logits_mode"] == (logprobs_mode == "raw_logits")
        return prompt_logprob.LogprobsTensors(
            logprob_token_ids=torch.zeros(
                prompt_logits.shape[0], num_logprobs + 1, dtype=torch.int64
            ),
            logprobs=torch.zeros(prompt_logits.shape[0], num_logprobs + 1),
            selected_token_ranks=torch.ones(prompt_logits.shape[0], dtype=torch.int64),
        )

    monkeypatch.setattr(prompt_logprob, "compute_distributed_topk_scores", distributed)
    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", fake_topk)
    token_ids, _, _ = prompt_logprob.compute_prompt_logprobs_with_chunking(
        torch.zeros(rows, dtype=torch.int64),
        torch.empty(rows, 1),
        gathered,
        local,
        num_prompt_logprobs,
        logprobs_mode,
    )

    assert token_ids.shape == (rows, num_prompt_logprobs + 1)
    assert calls.local_rows == calls.distributed_rows == [1024]
    assert calls.gathered_rows == [1]


def _tp2_asymmetric_worker(rank: int, init_method: str, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=init_method, world_size=2, rank=rank
    )
    try:
        prompt_logprob.get_tp_group = lambda: _GlooTPGroup
        full = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        if rank == 0:
            org = full[:, :4]
            indices = VocabParallelEmbeddingShardIndices(
                padded_org_vocab_start_index=0,
                padded_org_vocab_end_index=4,
                padded_added_vocab_start_index=5,
                padded_added_vocab_end_index=9,
                org_vocab_start_index=0,
                org_vocab_end_index=4,
                added_vocab_start_index=5,
                added_vocab_end_index=7,
            )
            tail_width = 4
        else:
            org = full[:, 4:]
            indices = VocabParallelEmbeddingShardIndices(
                padded_org_vocab_start_index=4,
                padded_org_vocab_end_index=8,
                padded_added_vocab_start_index=9,
                padded_added_vocab_end_index=13,
                org_vocab_start_index=4,
                org_vocab_end_index=5,
                added_vocab_start_index=7,
                added_vocab_end_index=7,
            )
            tail_width = 7
        local = torch.cat((org, torch.full((2, tail_width), 1e6)), dim=-1)
        result = prompt_logprob.compute_distributed_token_logprobs(
            LocalLogits(local, indices), torch.tensor([4, 1])
        )
        queue.put(
            (rank, result.logprobs[:, 0].tolist(), result.selected_token_ranks.tolist())
        )
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_asymmetric_org_shards_exclude_padded_and_added_tails(tmp_path):
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _tp2_asymmetric_worker,
        args=(f"file://{tmp_path / 'tp2-asymmetric-tail'}", queue),
        nprocs=2,
        join=True,
    )
    full = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    targets = torch.tensor([4, 1])
    expected_scores = torch.log_softmax(full, -1)[torch.arange(2), targets]
    expected_ranks = (full >= full[torch.arange(2), targets, None]).sum(-1)
    for _, scores, ranks in sorted(queue.get() for _ in range(2)):
        torch.testing.assert_close(torch.tensor(scores), expected_scores)
        assert ranks == expected_ranks.tolist()


def _tp2_topn_worker(rank: int, init_method: str, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=init_method, world_size=2, rank=rank
    )
    try:
        prompt_logprob.get_tp_group = lambda: _GlooTPGroup
        full = torch.tensor(
            [
                [-2.0, 1.0, 4.0, -1.0, 0.0, 2.0, 3.0],
                [5.0, -3.0, 0.5, 4.0, 2.0, 1.0, -2.0],
                [0.0, 3.0, -4.0, 1.0, 2.0, 5.0, -1.0],
            ]
        )
        start = rank * 4
        shard = full[:, start : start + 4]
        if rank == 1:
            shard = torch.cat((shard, torch.full((3, 1), 100.0)), dim=-1)
        indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=start + 4,
            padded_added_vocab_start_index=7,
            padded_added_vocab_end_index=8,
            org_vocab_start_index=start,
            org_vocab_end_index=min(start + 4, 7),
            added_vocab_start_index=7,
            added_vocab_end_index=8,
        )
        local = LocalLogits(shard, indices)
        targets = torch.tensor([6, 6, 0])
        results = []
        for num_logprobs in (1, 5):
            for mode in ("raw_logits", "raw_logprobs"):
                result = prompt_logprob.compute_distributed_topk_scores(
                    local, num_logprobs, targets, mode
                )
                results.append(
                    (
                        num_logprobs,
                        mode,
                        result.logprob_token_ids.tolist(),
                        result.logprobs.tolist(),
                        result.selected_token_ranks.tolist(),
                    )
                )
        queue.put((rank, results))
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_topn_raw_modes_padding_and_target_membership(tmp_path):
    """TP2 finite N=1/5 preserves gathered scores, ranks, and target-first IDs."""
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _tp2_topn_worker,
        args=(f"file://{tmp_path / 'tp2-topn'}", queue),
        nprocs=2,
        join=True,
    )
    full = torch.tensor(
        [
            [-2.0, 1.0, 4.0, -1.0, 0.0, 2.0, 3.0],
            [5.0, -3.0, 0.5, 4.0, 2.0, 1.0, -2.0],
            [0.0, 3.0, -4.0, 1.0, 2.0, 5.0, -1.0],
        ]
    )
    targets = torch.tensor([6, 6, 0])
    expected_ranks = (full >= full.gather(1, targets[:, None])).sum(-1)
    for _, results in sorted(queue.get() for _ in range(2)):
        for num_logprobs, mode, ids, scores, ranks in results:
            top_values, top_ids = torch.topk(full, num_logprobs, dim=-1)
            expected_ids = torch.cat((targets[:, None], top_ids), dim=1)
            expected_raw = torch.cat(
                (full.gather(1, targets[:, None]), top_values), dim=1
            )
            expected_scores = (
                expected_raw
                if mode == "raw_logits"
                else expected_raw - torch.logsumexp(full, dim=-1)[:, None]
            )
            assert ids == expected_ids.tolist()
            torch.testing.assert_close(torch.tensor(scores), expected_scores)
            assert ranks == expected_ranks.tolist()
            assert len(set(ids[0][1:])) == num_logprobs
            if num_logprobs == 5:
                assert any(target in row[1:] for target, row in zip(targets, ids))
                assert any(target not in row[1:] for target, row in zip(targets, ids))


def _tp2_topn_cutoff_tie_worker(rank: int, init_method: str, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=init_method, world_size=2, rank=rank
    )
    try:
        prompt_logprob.get_tp_group = lambda: _GlooTPGroup
        indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=rank * 3,
            padded_org_vocab_end_index=(rank + 1) * 3,
            padded_added_vocab_start_index=6,
            padded_added_vocab_end_index=6,
            org_vocab_start_index=rank * 3,
            org_vocab_end_index=(rank + 1) * 3,
            added_vocab_start_index=6,
            added_vocab_end_index=6,
        )
        local = LocalLogits(torch.zeros((2, 3)), indices)
        targets = torch.tensor([0, 5])
        results = []
        for num_logprobs in (1, 5):
            for mode in ("raw_logits", "raw_logprobs"):
                result = prompt_logprob.compute_distributed_topk_scores(
                    local, num_logprobs, targets, mode
                )
                results.append(
                    (
                        num_logprobs,
                        mode,
                        result.logprob_token_ids.tolist(),
                        result.logprobs.tolist(),
                        result.selected_token_ranks.tolist(),
                    )
                )
        queue.put((rank, results))
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_topn_allows_only_cutoff_tie_candidate_id_variation(tmp_path):
    """Cross-shard tied candidates keep exact target/rank/score semantics."""
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _tp2_topn_cutoff_tie_worker,
        args=(f"file://{tmp_path / 'tp2-topn-ties'}", queue),
        nprocs=2,
        join=True,
    )
    for _, results in sorted(queue.get() for _ in range(2)):
        for num_logprobs, mode, ids, scores, ranks in results:
            expected_score = (
                0.0 if mode == "raw_logits" else -torch.log(torch.tensor(6.0))
            )
            assert [row[0] for row in ids] == [0, 5]
            assert ranks == [6, 6]
            torch.testing.assert_close(
                torch.tensor(scores),
                torch.full((2, num_logprobs + 1), expected_score),
            )
            assert all(len(set(row[1:])) == num_logprobs for row in ids)


def _tp2_topn_mixed_cutoff_tie_worker(rank: int, init_method: str, queue) -> None:
    torch.distributed.init_process_group(
        backend="gloo", init_method=init_method, world_size=2, rank=rank
    )
    try:
        prompt_logprob.get_tp_group = lambda: _GlooTPGroup
        full = torch.tensor([[5.0, 4.0, 4.0, 4.0, 1.0, 0.0]])
        start = rank * 3
        indices = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=start + 3,
            padded_added_vocab_start_index=6,
            padded_added_vocab_end_index=6,
            org_vocab_start_index=start,
            org_vocab_end_index=start + 3,
            added_vocab_start_index=6,
            added_vocab_end_index=6,
        )
        local = LocalLogits(full[:, start : start + 3], indices)
        target = torch.tensor([5])
        results = []
        for mode in ("raw_logits", "raw_logprobs"):
            result = prompt_logprob.compute_distributed_topk_scores(
                local, 2, target, mode
            )
            results.append(
                (
                    mode,
                    result.logprob_token_ids.tolist(),
                    result.logprobs.tolist(),
                    result.selected_token_ranks.tolist(),
                )
            )
        queue.put((rank, results))
    finally:
        torch.distributed.destroy_process_group()


def test_tp2_topn_mixed_cutoff_tie_retains_strict_above_ids(tmp_path):
    """Only the equal-cutoff candidate may vary across the TP merge boundary."""
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()
    torch.multiprocessing.spawn(
        _tp2_topn_mixed_cutoff_tie_worker,
        args=(f"file://{tmp_path / 'tp2-topn-mixed-tie'}", queue),
        nprocs=2,
        join=True,
    )
    full = torch.tensor([[5.0, 4.0, 4.0, 4.0, 1.0, 0.0]])
    normalizer = torch.logsumexp(full, dim=-1)
    for _, results in sorted(queue.get() for _ in range(2)):
        for mode, ids, scores, ranks in results:
            assert ids[0][0] == 5
            assert ids[0][1] == 0  # The strictly-above-cutoff ID is mandatory.
            assert ids[0][2] in {1, 2, 3}
            assert len(set(ids[0][1:])) == 2
            expected_raw = torch.tensor([[0.0, 5.0, 4.0]])
            expected_scores = (
                expected_raw
                if mode == "raw_logits"
                else expected_raw - normalizer[:, None]
            )
            torch.testing.assert_close(torch.tensor(scores), expected_scores)
            assert ranks == [6]


def test_nvidia_dsa_local_logits_is_glm_only():
    from vllm.models.deepseek_v32.nvidia.model import DeepseekV32ForCausalLM

    class Processor:
        def __init__(self):
            self.calls = 0

        def get_local_logits(self, _: object, __: torch.Tensor) -> LocalLogits:
            self.calls += 1
            return _local_logits(torch.zeros(1, 2))

    processor = Processor()
    glm = SimpleNamespace(
        config=SimpleNamespace(model_type="glm_moe_dsa"),
        logits_processor=processor,
        lm_head=object(),
    )
    other = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v32"),
        logits_processor=processor,
        lm_head=object(),
    )
    assert DeepseekV32ForCausalLM.compute_local_logits(glm, torch.empty(1, 1))
    assert DeepseekV32ForCausalLM.compute_local_logits(other, torch.empty(1, 1)) is None
    assert processor.calls == 1


def test_dcp_falls_back_from_local_logits():
    class GLM:
        config = SimpleNamespace(model_type="glm_moe_dsa")

        @staticmethod
        def compute_local_logits(hidden_states: torch.Tensor) -> LocalLogits:
            return _local_logits(hidden_states)

    assert _get_prompt_logprobs_local_logits_fn(GLM(), dcp_size=1) is not None
    assert _get_prompt_logprobs_local_logits_fn(GLM(), dcp_size=4) is None
