# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute-side adapter for the optional independent UniEngram client.

Shared backing, checkpoint admission, connections and leases live in UniEngram.
Only output adaptation and vLLM execution checks belong here.
"""

from vllm.distributed.engram_share.graph import _validate_remote_graph_mode
from vllm.distributed.engram_share.lookup import EngramLookup
from vllm.distributed.engram_share.provider import EngramReaderOwner


def _open_client(checkpoint_path):
    try:
        from uniengram.cuda.checkpoint import CheckpointReader
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "uniengram":
            raise ImportError(
                "The UniEngram connector requires the independent "
                "uniengram-runtime package"
            ) from error
        raise
    return CheckpointReader(checkpoint_path)


def validate_config(config):
    _validate_remote_graph_mode(
        config.model_config.enforce_eager,
        config.compilation_config,
        config.engram_config.rdma_full_decode_graph,
    )


class UniEngramReader(EngramLookup):
    """Use independently managed raw rows with the existing vLLM scatter."""

    def __init__(self, checkpoint_path, *, full_decode_graph=False):
        client = _open_client(checkpoint_path)
        try:
            super().__init__(client, full_decode_graph=full_decode_graph)
        except BaseException:
            # Client close retains unsafe resources itself if retirement fails.
            client.close()
            raise


_reader_owner = EngramReaderOwner(UniEngramReader)
get_reader = _reader_owner.get_reader
close_reader = _reader_owner.close_reader
