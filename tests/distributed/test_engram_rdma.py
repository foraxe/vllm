# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM connector tests; owner/checkpoint tests live in UniEngram."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.distributed.engram_share import uniengram as rdma_provider
from vllm.distributed.engram_share.provider import get_provider


def test_connector_configuration_is_transport_neutral():
    from vllm.config.engram import EngramConfig

    config = EngramConfig(connector="external.mapped_tables")
    assert config.connector == "external.mapped_tables"
    assert config.rdma_provider == config.connector
    legacy = EngramConfig(rdma_provider="external.get_tables")
    assert legacy.connector == legacy.rdma_provider == "external.get_tables"
    with pytest.raises(ValueError, match="different modules"):
        EngramConfig(connector="external.one", rdma_provider="external.two")


def test_rdma_rejects_packed_fp4_before_allocation():
    from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding

    with pytest.raises(ValueError, match="rdma_provider.*packed FP4"):
        ParallelEngramEmbedding(1, 256, (1,), rdma_provider="test", packed_fp4=True)


def test_external_provider_is_initialized_only_by_the_loader(tmp_path, monkeypatch):
    """Loading the interface does not open a remote reader or acquire a lease."""
    (tmp_path / "test_engram_provider.py").write_text(
        "reader = None\n"
        "def get_reader(checkpoint_path=None):\n"
        "    global reader\n"
        "    if reader is None:\n"
        "        if checkpoint_path is None: raise RuntimeError('not initialized')\n"
        "        reader = checkpoint_path\n"
        "    return reader\n"
        "def close_reader():\n"
        "    global reader\n"
        "    reader = None\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    provider = get_provider("test_engram_provider")
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.get_reader()
    assert provider.get_reader("checkpoint") == "checkpoint"
    assert provider.get_reader() == "checkpoint"
    provider.close_reader()
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.get_reader()


@pytest.mark.parametrize("retirement_fails", [False, True])
def test_reader_setup_failure_retires_or_quarantines_runtime(
    tmp_path, monkeypatch, retirement_fails
):
    """A CUDA event failure must not release unsafe registered resources."""
    runtimes = []

    class Runtime:
        storage_layout = "row264"

        def __init__(self, checkpoint_path):
            self.closed = False
            runtimes.append(self)

        def close(self):
            if retirement_fails:
                raise RuntimeError("retirement failed")
            self.closed = True

    def fail_event():
        raise RuntimeError("event creation failed")

    monkeypatch.setattr(rdma_provider, "_open_client", Runtime)
    monkeypatch.setattr(torch.cuda, "Event", fail_event)
    with pytest.raises(
        RuntimeError,
        match="retirement failed" if retirement_fails else "event creation failed",
    ):
        rdma_provider.UniEngramReader(tmp_path)
    assert runtimes[0].closed is not retirement_fails


def test_gpu_get_adapter_keeps_vllm_row_semantics():
    """Real GPU adapter/kernel with an external row-service substitute, not NIC GET.

    Run in a child so the component script's injected transport cannot leak into
    the checkpoint/owner tests. The normal distributed CI step allocates GPUs.
    """
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            str(root / "examples/disaggregated/engram_rdma/test_provider_gpu.py"),
        ],
        cwd=root,
        env=env,
        check=True,
        timeout=120,
    )
