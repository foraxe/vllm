# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mapped reader view requirements for the existing Engram kernel."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def provider():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm/distributed/engram_share/provider.py"
    )
    spec = importlib.util.spec_from_file_location("engram_provider_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "invalid",
    [None, "weight_stride", "scale_stride", "weight_alignment", "scale_alignment"],
)
def test_mapped_views_reject_kernel_incompatible_layout(provider, invalid):
    """Real tensor views can have valid shapes but unsafe row strides/alignment."""
    import torch

    class Layout:
        dim = 256
        block_size = 32
        vocab_start_idx = 3
        vocab_end_idx = 5

    weight = torch.empty(2, 256, dtype=torch.float8_e4m3fn)
    scale = torch.empty(2, 8, dtype=torch.uint8)
    if invalid == "weight_stride":
        weight = torch.empty(2, 257, dtype=weight.dtype)[:, :256]
    elif invalid == "scale_stride":
        scale = torch.empty(2, 9, dtype=scale.dtype)[:, :8]
    elif invalid == "weight_alignment":
        weight = torch.empty(513, dtype=torch.uint8)[1:].view(weight.dtype).view(2, 256)
    elif invalid == "scale_alignment":
        scale = torch.empty(17, dtype=torch.uint8)[1:].view(2, 8)

    class MappedBackend:
        def storage(self, table):
            return weight, scale

    if invalid is None:
        actual_weight, actual_scale = provider.mapped_storage(MappedBackend(), Layout())
        assert actual_weight is weight and actual_scale is scale
    else:
        with pytest.raises(ValueError, match="aligned contiguous"):
            provider.mapped_storage(MappedBackend(), Layout())
