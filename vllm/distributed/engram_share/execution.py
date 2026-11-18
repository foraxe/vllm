# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional provider execution scope around framework-owned graph launches."""

from functools import wraps

from vllm.distributed.engram_share.provider import get_provider


def with_provider_execution(func):
    @wraps(func)
    def run(self, *args, **kwargs):
        config = self.vllm_config.engram_config
        if config is None or not config.rdma_provider:
            return func(self, *args, **kwargs)
        provider = get_provider(config.rdma_provider)
        scope = getattr(provider, "execution_scope", None)
        if scope is None:
            return func(self, *args, **kwargs)
        with scope():
            return func(self, *args, **kwargs)

    return run
