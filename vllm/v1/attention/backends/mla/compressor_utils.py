# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher, zip_inputs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_scalar_specialization_rep,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

_DSPARK_SWA_INDEX_ALIGNMENT = 64


def get_dspark_swa_index_width(
    window_size: int,
    num_speculative_tokens: int,
) -> int:
    """Return the padded width of non-causal DSpark SWA indices."""
    width = max(int(window_size), 0) + max(int(num_speculative_tokens), 0)
    return cdiv(width, _DSPARK_SWA_INDEX_ALIGNMENT) * _DSPARK_SWA_INDEX_ALIGNMENT


class CompressedSlotMappingKernel(
    VllmTritonJitKernel["CompressedSlotMappingKernel.CompileKey"]
):
    TRITON_BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        compress_ratio: int
        triton_block_size: int
        block_size: int
        dcp_rank: int
        dcp_world_size: int
        dcp_interleave: int

    @staticmethod
    @triton.jit(do_not_specialize=["block_table_stride"])
    def kernel(
        # [num_tokens]
        slot_mapping_ptr,
        # [num_reqs + 1]
        query_start_loc_ptr,
        # [num_reqs]
        seq_lens_ptr,
        # [num_reqs, max_num_blocks]
        block_table_ptr,
        block_table_stride,
        block_size,
        DCP_RANK: tl.constexpr,
        DCP_WORLD_SIZE: tl.constexpr,
        DCP_INTERLEAVE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
        PAD_ID: tl.constexpr,
        TRITON_BLOCK_SIZE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        start_pos = seq_len - query_len

        for i in range(0, query_len, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            mask = offset < query_len

            pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
            is_valid = (pos + 1) % COMPRESS_RATIO == 0
            global_record = pos // COMPRESS_RATIO
            owner = (global_record // DCP_INTERLEAVE) % DCP_WORLD_SIZE
            group_round = global_record // (DCP_INTERLEAVE * DCP_WORLD_SIZE)
            group_offset = global_record % DCP_INTERLEAVE
            pos_after_compress = group_round * DCP_INTERLEAVE + group_offset
            is_valid &= owner == DCP_RANK

            block_ids = pos_after_compress // block_size
            block_numbers = tl.load(
                block_table_ptr + batch_idx * block_table_stride + block_ids,
                mask=mask & is_valid,
            )
            slot_ids = block_numbers * block_size + pos_after_compress % block_size

            # NOTE
            slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
            tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        compress_ratio: int,
        block_size: int,
        dcp_rank: int = 0,
        dcp_world_size: int = 1,
        dcp_interleave: int = 1,
    ) -> CompileKey:
        return self.CompileKey(
            compress_ratio=compress_ratio,
            triton_block_size=self.TRITON_BLOCK_SIZE,
            block_size=triton_scalar_specialization_rep(block_size),
            dcp_rank=triton_scalar_specialization_rep(dcp_rank),
            dcp_world_size=triton_scalar_specialization_rep(dcp_world_size),
            dcp_interleave=triton_scalar_specialization_rep(dcp_interleave),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_text_config = vllm_config.model_config.hf_text_config
        configured_ratios = (
            *(getattr(hf_text_config, "compress_ratios", None) or ()),
            getattr(hf_text_config, "index_kpool", 1) or 1,
        )
        compress_ratios = tuple(
            dict.fromkeys(int(ratio) for ratio in configured_ratios if int(ratio) > 1)
        )
        if not compress_ratios:
            return []
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dcp_world_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        dcp_rank = 0
        dcp_interleave = getattr(parallel_config, "cp_kv_cache_interleave_size", 1)
        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                *(
                    dict(
                        compress_ratio=ratio,
                        block_size=vllm_config.cache_config.block_size // ratio,
                        dcp_rank=dcp_rank,
                        dcp_world_size=dcp_world_size,
                        dcp_interleave=dcp_interleave,
                    )
                    for ratio in compress_ratios
                )
            )
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        int32_ptr = TritonWarmupTensor(torch.int32)
        return dict(
            slot_mapping=TritonWarmupTensor(torch.int64),
            query_start_loc=int32_ptr,
            seq_lens=int32_ptr,
            block_table=int32_ptr,
            block_size=compile_key.block_size,
            compress_ratio=compile_key.compress_ratio,
            dcp_rank=compile_key.dcp_rank,
            dcp_world_size=compile_key.dcp_world_size,
            dcp_interleave=compile_key.dcp_interleave,
        )

    @kernel_launcher
    def __call__(
        self,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        compress_ratio: int,
        dcp_rank: int = 0,
        dcp_world_size: int = 1,
        dcp_interleave: int = 1,
    ) -> LaunchSpec:
        return (block_table.shape[0],), dict(
            block_table_stride=block_table.stride(0),
            DCP_RANK=dcp_rank,
            DCP_WORLD_SIZE=dcp_world_size,
            DCP_INTERLEAVE=dcp_interleave,
            COMPRESS_RATIO=compress_ratio,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=self.TRITON_BLOCK_SIZE,
        )


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    dcp_interleave: int = 1,
) -> torch.Tensor:
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    _COMPRESSED_SLOT_MAPPING_KERNEL(
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        dcp_rank,
        dcp_world_size,
        dcp_interleave,
    )
    return slot_mapping


def get_compressed_record_owner_and_local(
    position: int,
    compress_ratio: int,
    dcp_world_size: int,
    dcp_interleave: int = 1,
) -> tuple[int, int] | None:
    """Return the owner and owner-local record for an emitted raw position."""
    if (position + 1) % compress_ratio != 0:
        return None
    record = position // compress_ratio
    owner = (record // dcp_interleave) % dcp_world_size
    local = (
        record // (dcp_interleave * dcp_world_size) * dcp_interleave
        + record % dcp_interleave
    )
    return owner, local


_COMPRESSED_SLOT_MAPPING_KERNEL = CompressedSlotMappingKernel()
