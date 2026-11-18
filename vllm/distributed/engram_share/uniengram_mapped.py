# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional UniEngram mapped checkpoint connector."""

from vllm.distributed.engram_share.provider import EngramReaderOwner


def validate_config(config):
    """Mapped-reader capability checks are performed by the external client."""


def _open_client(checkpoint_path, *, full_decode_graph=False):
    try:
        from uniengram.cuda.mapped_checkpoint import MappedCheckpointReader
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == "uniengram":
            raise ImportError(
                "The mapped UniEngram connector requires the independent "
                "uniengram-runtime package with mapped checkpoint support"
            ) from error
        raise
    return MappedCheckpointReader(checkpoint_path)


_reader_owner = EngramReaderOwner(_open_client)
get_reader = _reader_owner.get_reader
close_reader = _reader_owner.close_reader
