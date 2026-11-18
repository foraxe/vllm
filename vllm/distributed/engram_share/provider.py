# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-local ownership and access contracts for external Engram tables."""

import importlib
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig


class EngramTable(Protocol):
    """Model layout and local head range, independent of storage/transport."""

    dim: int
    block_size: int
    num_embeddings: int
    head_sizes: tuple[int, ...]
    head_start: int
    part_n_hash_cols: int
    vocab_start_idx: int
    vocab_end_idx: int
    layer_id: int | None


class EngramReader(Protocol):
    @property
    def access_mode(self) -> Literal["lookup", "mapped"]: ...

    def close(self) -> None:
        """Retire device access before releasing backing or owner leases."""
        ...


class EngramLookupReader(EngramReader, Protocol):
    def lookup(
        self,
        table: EngramTable,
        indices: "torch.Tensor",
        out: "torch.Tensor",
        background: bool = False,
    ) -> None:
        """Write the model's output on the current stream; retain internal scratch."""
        ...


class EngramMappedReader(EngramReader, Protocol):
    uses_small_host_pages: bool

    def storage(self, table: EngramTable) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return stable local-head views for the existing Engram lookup kernel.

        This NVIDIA model seam uses weight and scale tensors. It does not claim
        that arbitrary multi-owner mappings or native CANN chunk pointers are
        contiguous tensors. The reader retains all mappings until close.
        """
        ...


def mapped_storage(
    reader: EngramMappedReader, table: EngramTable
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Validate views against the existing kernel's tight, aligned row layout."""
    import torch

    weight, scale = reader.storage(table)
    rows = table.vocab_end_idx - table.vocab_start_idx
    if (
        weight.shape != (rows, table.dim)
        or scale.shape != (rows, table.dim // table.block_size)
        or weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.uint8
        or weight.device != scale.device
        or not weight.is_contiguous()
        or not scale.is_contiguous()
        or weight.data_ptr() % 8
        or scale.data_ptr() % 8
    ):
        raise ValueError(
            "mapped Engram views require the model's aligned contiguous layout"
        )
    return weight, scale


class EngramProvider(Protocol):
    def validate_config(self, config: "VllmConfig") -> None:
        """Check actual execution settings without opening a reader."""
        ...

    def get_reader(
        self,
        checkpoint_path: str | None = None,
    ) -> EngramReader: ...

    def close_reader(self) -> None: ...


def get_provider(module_name: str) -> EngramProvider:
    """Import a provider without creating a reader, mapping or owner lease."""
    return cast(EngramProvider, importlib.import_module(module_name))
