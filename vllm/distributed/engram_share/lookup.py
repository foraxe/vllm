# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Engram lookup adaptation; transport and owner control stay in backends."""

from typing import Literal, Protocol

import torch
import torch.nn.functional as F

from vllm.distributed.engram_share.provider import EngramTable
from vllm.triton_utils import tl, triton


class EngramRowReader(Protocol):
    def fetch(
        self, ids: torch.Tensor, layer: int | None, lo: int, hi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def record_consumed(self, layer: int | None) -> None: ...

    def close(self) -> None: ...


@triton.jit
def _scatter_rows(
    weight,
    scales,
    ids,
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
    row = tl.program_id(0)
    idx = tl.load(ids + row).to(tl.int64)
    owned = (idx >= lo) & (idx < hi)
    cols = tl.arange(0, 256)
    values = tl.load(weight + row * stride_w + cols, mask=owned, other=0.0)
    scale = tl.load(scales + row * stride_s + cols // 32, mask=owned, other=0)
    dst = out + (row // local_heads) * stride_t + (row % local_heads) * stride_h
    if packed:
        tl.store(dst + cols, values.to(tl.uint8, bitcast=True))
        tl.store(dst + 256 + cols // 32, scale, mask=cols % 32 == 0)
    else:
        factor = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(dst + cols, (values.to(tl.float32) * factor).to(tl.bfloat16))


class EngramLookup:
    """Adapt raw GPU row reads to the existing Engram output/kernel semantics."""

    access_mode: Literal["lookup"] = "lookup"

    def __init__(self, runtime: EngramRowReader, *, full_decode_graph: bool = False):
        self.runtime = runtime
        self.full_decode_graph = full_decode_graph
        self.consumed = torch.cuda.Event()
        self.inflight = False
        self.last_stream = None
        self.captured_stream = None

    def lookup(
        self,
        table: EngramTable,
        indices: torch.Tensor,
        out: torch.Tensor,
        background: bool = False,
    ) -> None:
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not self.full_decode_graph:
            raise RuntimeError("remote lookup must run at a vLLM eager boundary")
        if table.dim != 256 or table.block_size != 32:
            raise ValueError("unsupported remote Engram row format")
        if out.dtype not in (torch.bfloat16, torch.uint8) or out.ndim != 3:
            raise ValueError("expected 3-D BF16 or packed-byte output")
        if out.shape[-1] != (264 if out.dtype == torch.uint8 else 256):
            raise ValueError("output row width does not match its dtype")
        if out.shape[:2] != (indices.shape[0], table.part_n_hash_cols):
            raise ValueError("output shape does not match local Engram heads")
        start, heads = table.head_start, table.part_n_hash_cols
        local = indices[:, start : start + heads]
        if local.shape[1] < heads:
            local = F.pad(local, (0, heads - local.shape[1]), value=-1)
        ids = local.to(dtype=torch.int64).contiguous().view(-1)
        # The native GET validates every input ID before applying rank bounds.
        # vLLM uses negative IDs for padding, so give the transport a valid
        # placeholder and retain the original IDs for the final ownership mask.
        read_ids = torch.where((ids >= 0) & (ids < table.num_embeddings), ids, 0)
        stream = torch.cuda.current_stream()
        if (
            self.captured_stream is not None
            and self.captured_stream != stream.cuda_stream
        ):
            raise RuntimeError(
                "captured remote reader must stay on its prefetch stream"
            )
        if self.inflight:
            if capturing and self.last_stream != stream.cuda_stream:
                raise RuntimeError("warm remote reader on capture stream")
            # An event recorded during capture cannot be waited on by unrelated
            # eager work. The same stream already orders reuse of the arena.
            if not capturing and self.last_stream != stream.cuda_stream:
                stream.wait_event(self.consumed)
        _, weight, scale = self.runtime.fetch(
            read_ids, table.layer_id, table.vocab_start_idx, table.vocab_end_idx
        )
        _scatter_rows[(ids.numel(),)](
            weight,
            scale,
            ids,
            out,
            table.vocab_start_idx,
            table.vocab_end_idx,
            weight.stride(0),
            scale.stride(0),
            out.stride(0),
            out.stride(1),
            heads,
            out.dtype == torch.uint8,
            num_warps=4,
        )
        self.runtime.record_consumed(table.layer_id)
        self.consumed.record(stream)
        self.inflight = True
        self.last_stream = stream.cuda_stream
        if capturing:
            self.captured_stream = stream.cuda_stream

    def close(self) -> None:
        """Retire GPU and NIC work before returning the owner lease."""
        self.runtime.close()
