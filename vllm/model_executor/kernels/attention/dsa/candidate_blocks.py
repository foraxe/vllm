# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.triton_utils import tl, triton

DCP_BLOCK_SCORE_WORKSPACE_BYTES = 64 * 1024 * 1024


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "block_start", "block_count"])
def _dcp_block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    nan_flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    block_start,
    block_count,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = block_start + tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    global_records = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    owned = (global_records % DCP_WORLD_SIZE) == DCP_RANK
    local_records = global_records // DCP_WORLD_SIZE
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    cols = start + local_records
    valid = (
        (blocks[:, None] < block_start + block_count)
        & (offsets[None, :] < BLOCK_SIZE)
        & owned
        & (cols < end)
        & (cols < width)
    )
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        valid,
        other=-float("inf"),
    )
    has_nan = tl.max(tl.where(valid & (values != values), 1, 0), axis=1)
    values = tl.where(values != values, -float("inf"), values)
    reduced = tl.max(values, axis=1)
    out_blocks = blocks - block_start
    store_mask = blocks < block_start + block_count
    tl.store(scores + row * block_count + out_blocks, reduced, store_mask)
    tl.store(nan_flags + row * block_count + out_blocks, has_nan, store_mask)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value > -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    if DCP_WORLD_SIZE == 1:
        # Preserve the packed-column clamp for candidates beyond the logits width.
        block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    else:
        block = tl.where(block >= nblocks, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    record = cols - start
    if DCP_WORLD_SIZE > 1:
        record = record * DCP_WORLD_SIZE + DCP_RANK
    block = record // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    if DCP_WORLD_SIZE == 1:
        edge = tl.load(flags + row * (nblocks + 1) + nblocks)
        keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    else:
        keep = keep != 0
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def merge_dcp_block_score_shards(
    score_shards: torch.Tensor,
    nan_shards: torch.Tensor,
    global_row_lens: torch.Tensor,
    block_start: int,
    topk_blocks: int,
    block_size: int,
    prior_scores: torch.Tensor | None = None,
    prior_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one global-block chunk and retain the best candidates so far."""
    assert score_shards.ndim == 3 and nan_shards.shape == score_shards.shape
    scores = score_shards.amax(dim=0)
    scores.masked_fill_(nan_shards.amax(dim=0).bool(), float("nan"))
    rows, block_count = scores.shape
    ids = torch.arange(
        block_start,
        block_start + block_count,
        device=scores.device,
        dtype=torch.int64,
    ).expand(rows, -1)
    newest = torch.div(global_row_lens - 1, block_size, rounding_mode="floor")
    scores.masked_fill_(
        (global_row_lens > 0)[:, None] & (ids == newest[:, None]), float("inf")
    )
    if prior_scores is not None:
        assert prior_ids is not None
        scores = torch.cat((prior_scores, scores), dim=1)
        ids = torch.cat((prior_ids, ids), dim=1)
    keep = min(topk_blocks, scores.shape[1])
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :keep]
    kept_scores = scores.gather(1, order)
    kept_ids = ids.gather(1, order)
    return kept_scores, kept_ids


def select_dcp_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    global_row_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    gather: Callable[..., torch.Tensor],
    row_repeat: int = 1,
    chunk_blocks: int = 4096,
) -> None:
    """Select global blocks from record-striped DCP logits with bounded scratch."""
    assert logits.is_cuda
    assert dcp_world_size in (2, 4)
    assert topk_blocks > 0 and block_size > 0 and chunk_blocks > 0
    rows, width = logits.shape
    if not rows:
        return
    bytes_per_block = rows * (dcp_world_size + 1) * 5
    chunk_blocks = min(
        chunk_blocks,
        max(1, DCP_BLOCK_SCORE_WORKSPACE_BYTES // bytes_per_block),
    )
    max_global_len = int(global_row_lens.max().item()) if rows else 0
    total_blocks = triton.cdiv(max_global_len, block_size)
    if total_blocks == 0:
        out.fill_(-1)
        return
    running_scores = None
    running_ids = None
    for block_start in range(0, total_blocks, chunk_blocks):
        block_count = min(chunk_blocks, total_blocks - block_start)
        scores = logits.new_empty((rows, block_count))
        nan_flags = torch.empty(
            (rows, block_count), dtype=torch.uint8, device=logits.device
        )
        _dcp_block_scores_kernel[(rows, triton.cdiv(block_count, 128))](
            logits,
            row_ks,
            row_ke,
            scores,
            nan_flags,
            *logits.stride(),
            row_ks.stride(0) if row_ks is not None else 0,
            row_ke.stride(0),
            width,
            block_start,
            block_count,
            dcp_rank,
            dcp_world_size,
            block_size,
            row_ks is not None,
            row_repeat,
            128,
        )
        gathered_scores = gather(scores, dim=0).reshape(
            dcp_world_size, rows, block_count
        )
        gathered_nans = gather(nan_flags, dim=0).reshape(
            dcp_world_size, rows, block_count
        )
        running_scores, running_ids = merge_dcp_block_score_shards(
            gathered_scores,
            gathered_nans,
            global_row_lens,
            block_start,
            topk_blocks,
            block_size,
            running_scores,
            running_ids,
        )
    assert running_scores is not None and running_ids is not None
    out.fill_(-1)
    valid = running_scores > -float("inf")
    out[:, : running_ids.shape[1]].copy_(
        torch.where(valid, running_ids, -1).to(out.dtype)
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    global_max_len: int | None = None,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    if dcp_world_size > 1:
        assert dcp_world_size in (2, 4)
        assert global_max_len is not None
        nblocks = triton.cdiv(global_max_len, block_size)
    else:
        nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
        dcp_world_size,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        dcp_rank,
        dcp_world_size,
        1024,
    )
