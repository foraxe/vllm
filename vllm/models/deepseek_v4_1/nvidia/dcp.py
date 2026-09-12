# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit-communication helpers for DeepSeek V4.1 DCP attention."""

import torch

from vllm.distributed import get_dcp_group


def localize_topk(
    indices: torch.Tensor, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep this owner's record IDs, compacting holes before paged lookup."""
    valid = (indices >= 0) & (indices % world_size == rank)
    width = indices.shape[-1]
    positions = torch.arange(width, device=indices.device)
    order = torch.where(valid, positions, width).argsort(dim=-1)
    local = torch.where(valid, indices // world_size, -1)
    return local.gather(-1, order), valid.sum(dim=-1, dtype=torch.int32)


def gather_query(q: torch.Tensor, local_heads: int) -> torch.Tensor:
    """Gather real TP heads, then restore the native kernel's head padding."""
    gathered = get_dcp_group().all_gather(q[:, :local_heads].contiguous(), dim=1)
    heads = gathered.shape[1]
    padded = 64 if heads <= 64 else 128
    if heads > padded:
        raise ValueError(f"DeepSeek V4.1 DCP requires at most 128 heads, got {heads}")
    return torch.nn.functional.pad(gathered, (0, 0, 0, padded - heads))


def merge_partial_output(
    output: torch.Tensor,
    lse: torch.Tensor,
    valid_counts: torch.Tensor,
    local_heads: int,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge unsunk partials and apply the sink once on the query-head owner.

    FlashMLA returns +inf LSE for empty rows and excludes the sink from LSE.
    Metadata counts, rather than the LSE sentinel, identify empty partitions.
    """
    group = get_dcp_group()
    valid = valid_counts[:, None] > 0
    lse = torch.where(valid, lse, -torch.inf)
    partial = torch.where(valid[..., None], output.float(), 0.0)
    all_lse = group.all_gather(lse.contiguous(), dim=0).view(
        group.world_size, *lse.shape
    )
    total_lse = torch.logsumexp(all_lse, dim=0)
    weight = torch.exp(torch.where(valid, lse - total_lse, -torch.inf))
    merged = group.all_reduce((partial * weight[..., None]).contiguous())
    first = group.rank_in_group * local_heads
    local_lse = total_lse[:, first : first + local_heads]
    denominator = torch.logaddexp(local_lse, sink[None, :local_heads])
    factor = torch.where(
        torch.isneginf(local_lse), 0.0, torch.exp(local_lse - denominator)
    )
    return merged[:, first : first + local_heads] * factor[..., None]


def prefill_indices(
    topk: torch.Tensor,
    positions: torch.Tensor,
    request_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    ratio: int,
    window: int,
    request_stride: int,
    swa_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index gathered owner-local records and one copy of the replicated SWA.

    Requests occupy separate rows of the gathered buffer. Preserve -1 sentinels
    before adding each request's base, including for requests after the first.
    """
    requests = request_indices.long()
    valid = (topk >= 0) & (topk % world_size == rank)
    valid &= topk < ((positions[:, None] + 1) // ratio)
    base = requests[:, None] * request_stride
    main = torch.where(valid, base + topk // world_size, -1)
    offsets = torch.arange(window, device=topk.device)
    first = (positions - window + 1).clamp_min(0)
    swa_positions = first[:, None] + offsets
    swa_valid = (swa_positions <= positions[:, None]) & (rank == 0)
    gather_start = seq_lens[requests] - gather_lens[requests]
    swa = torch.where(
        swa_valid, base + swa_offset + swa_positions - gather_start[:, None], -1
    )
    indices = torch.cat((main, swa), dim=-1).to(torch.int32)
    padded = ((indices.shape[1] + 127) // 128) * 128
    indices = torch.nn.functional.pad(indices, (0, padded - indices.shape[1]), value=-1)
    return indices, (indices >= 0).sum(dim=-1, dtype=torch.int32)
