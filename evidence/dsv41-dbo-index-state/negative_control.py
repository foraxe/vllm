# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.worker import ubatching

# Reproduce the incumbent ownership: both microbatches use the same storage.
ubatching.dbo_select_buffer = lambda buffer: buffer[0] if buffer.ndim == 3 else buffer
raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/v1/worker/test_gpu_ubatch_slicing.py",
            "-k",
            "preserves_cross_layer_index_state",
        ]
    )
)
