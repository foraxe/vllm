# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU query compaction and packed-row scatter for the optional gather path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _representatives(ids, table, reps, work, lo, hi, S: tl.constexpr):
    q = tl.program_id(0).to(tl.int64)
    row = tl.load(ids + q).to(tl.int64)
    representative = tl.full((), -1, tl.int64)
    if (row >= lo) & (row < hi):
        slot = (row * 2654435761) & (S - 1)
        key = ((row.to(tl.uint64) + 1) << 32) | q.to(tl.uint64)
        seen = tl.atomic_cas(
            table + slot, tl.full((), 0, tl.uint64), key, sem="relaxed"
        )
        while (seen != 0) & ((seen >> 32) != (row.to(tl.uint64) + 1)):
            slot = (slot + 1) & (S - 1)
            seen = tl.atomic_cas(
                table + slot, tl.full((), 0, tl.uint64), key, sem="relaxed"
            )
        representative = tl.where(seen == 0, q, (seen & 0xFFFFFFFF).to(tl.int64))
        if seen == 0:
            pos = tl.atomic_add(table + S, tl.full((), 1, tl.uint64), sem="relaxed")
            tl.store(work + pos, q)
    tl.store(reps + q, representative)


def prepare(ids, lo, hi, scratch, reps, work):
    if not 0 <= lo <= hi < 2**32 or ids.numel() >= 2**32:
        raise ValueError("gather ID range exceeds compaction encoding")
    scratch.zero_()
    if ids.numel():
        _representatives[(ids.numel(),)](
            ids, scratch, reps, work, lo, hi, S=scratch.numel() - 1, num_warps=1
        )


@triton.jit
def _scatter_mapped(
    weight,
    scales,
    ids,
    dense,
    out,
    lo,
    hi,
    stride_w,
    stride_s,
    stride_t,
    stride_h,
    local_heads: tl.constexpr,
    packed: tl.constexpr,
):
    q = tl.program_id(0)
    idx = tl.load(ids + q).to(tl.int64)
    slot = tl.load(dense + q).to(tl.int64)
    owned = (idx >= lo) & (idx < hi) & (slot >= 0) & (slot < tl.num_programs(0))
    cols = tl.arange(0, 256)
    values = tl.load(weight + slot * stride_w + cols, mask=owned, other=0.0)
    scale = tl.load(scales + slot * stride_s + cols // 32, mask=owned, other=0)
    dst = out + (q // local_heads) * stride_t + (q % local_heads) * stride_h
    if packed:
        tl.store(dst + cols, values.to(tl.uint8, bitcast=True))
        tl.store(dst + 256 + cols // 32, scale, mask=cols % 32 == 0)
    else:
        factor = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(dst + cols, (values.to(tl.float32) * factor).to(tl.bfloat16))


def workspace(maximum, device):
    slots = triton.next_power_of_2(max(2, 2 * maximum))
    return {
        "scratch": torch.empty(slots + 1, dtype=torch.uint64, device=device),
        "reps": torch.empty(maximum, dtype=torch.int64, device=device),
        "work": torch.empty(maximum, dtype=torch.int64, device=device),
        "dense": torch.empty(maximum, dtype=torch.int64, device=device),
        "inverse": torch.empty(maximum, dtype=torch.int64, device=device),
    }
