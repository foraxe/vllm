# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execution checks for the external GPU lookup adapter."""

from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config.compilation import CompilationConfig, CUDAGraphMode


def _validate_remote_graph_mode(
    enforce_eager: bool,
    compilation_config: CompilationConfig,
    full_decode: bool = False,
) -> None:
    mode = compilation_config.cudagraph_mode
    breakable = is_breakable_cudagraph_enabled(compilation_config)
    allowed_full = full_decode and (
        mode == CUDAGraphMode.FULL_DECODE_ONLY
        or (mode == CUDAGraphMode.FULL_AND_PIECEWISE and breakable)
    )
    if (
        not enforce_eager
        and not allowed_full
        and (mode != CUDAGraphMode.PIECEWISE or not breakable)
    ):
        raise ValueError(
            "rdma_provider requires enforce_eager, breakable PIECEWISE, or "
            "explicitly enabled FULL decode CUDA graphs"
        )
