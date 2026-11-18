# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker ownership contract, using actual provider code and CPU-only backends."""

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


class Backend:
    def __init__(self, mode):
        self.access_mode = mode
        self.close_calls = 0
        self.fail_close = False
        self.payload = object()

    def lookup(self, *args):
        return self.payload

    def storage(self, *args):
        return self.payload, self.payload

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("retirement failed")


@pytest.mark.parametrize("mode", ["mapped", "lookup"])
def test_loader_owns_one_reader_and_can_reattach(provider, tmp_path, mode):
    created = []

    def factory(checkpoint, *, full_decode_graph):
        result = Backend(mode)
        created.append((checkpoint, full_decode_graph, result))
        return result

    owner = provider.EngramReaderOwner(factory)
    with pytest.raises(RuntimeError, match="model loader"):
        owner.get_reader()
    first = owner.get_reader(str(tmp_path / "."), full_decode_graph=True)
    assert owner.get_reader() is first
    assert owner.get_reader(str(tmp_path), full_decode_graph=True) is first
    assert len(created) == 1 and created[0][:2] == (str(tmp_path.resolve()), True)
    if mode == "mapped":
        assert first.storage(None) == (first.payload, first.payload)
    else:
        assert first.lookup(None, None, None) is first.payload
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different engine checkpoint"):
        owner.get_reader(str(other))
    with pytest.raises(ValueError, match="graph configuration"):
        owner.get_reader(full_decode_graph=False)
    owner.close_reader()
    assert first.close_calls == 1
    with pytest.raises(RuntimeError, match="model loader"):
        owner.get_reader()
    second = owner.get_reader(str(other), full_decode_graph=False)
    assert second is not first and len(created) == 2
    owner.close_reader()
    assert second.close_calls == 1


def test_factory_failure_allows_retry(provider, tmp_path):
    calls = []

    def factory(checkpoint, *, full_decode_graph):
        calls.append(checkpoint)
        if len(calls) == 1:
            raise RuntimeError("attach failed")
        return Backend("lookup")

    owner = provider.EngramReaderOwner(factory)
    with pytest.raises(RuntimeError, match="attach failed"):
        owner.get_reader(str(tmp_path))
    reader = owner.get_reader(str(tmp_path))
    assert owner.get_reader() is reader and len(calls) == 2
    owner.close_reader()


def test_close_failure_blocks_borrow_and_retains_reader_for_retry(provider, tmp_path):
    backend = Backend("mapped")
    backend.fail_close = True
    created = []

    def factory(*args, **kwargs):
        created.append(backend)
        return backend

    owner = provider.EngramReaderOwner(factory)
    assert owner.get_reader(str(tmp_path)) is backend
    with pytest.raises(RuntimeError, match="retirement failed"):
        owner.close_reader()
    with pytest.raises(RuntimeError, match="closing"):
        owner.get_reader(str(tmp_path))
    assert (
        len(created) == 1 and backend.close_calls == 1 and backend.payload is not None
    )
    backend.fail_close = False
    owner.close_reader()
    assert backend.close_calls == 2
    assert owner.get_reader(str(tmp_path)) is backend
    assert len(created) == 2
    owner.close_reader()


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


@pytest.mark.parametrize("retirement_fails", [False, True])
def test_invalid_access_mode_is_retired_or_retained(
    provider, tmp_path, retirement_fails
):
    backend = Backend("unknown")
    backend.fail_close = retirement_fails
    owner = provider.EngramReaderOwner(lambda *args, **kwargs: backend)
    if retirement_fails:
        with pytest.raises(RuntimeError, match="retirement failed"):
            owner.get_reader(str(tmp_path))
        with pytest.raises(RuntimeError, match="closing"):
            owner.get_reader()
        backend.fail_close = False
        owner.close_reader()
    else:
        with pytest.raises(ValueError, match="access mode"):
            owner.get_reader(str(tmp_path))
    assert backend.close_calls == (2 if retirement_fails else 1)
    with pytest.raises(RuntimeError, match="model loader"):
        owner.get_reader()


def test_mapped_connector_owns_external_attachment(tmp_path, monkeypatch):
    """The optional client owns one mapping until the worker closes it."""
    import importlib

    package = tmp_path / "uniengram" / "cuda"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "mapped_checkpoint.py").write_text(
        "class MappedCheckpointReader:\n"
        "    access_mode = 'mapped'\n"
        "    def __init__(self, checkpoint):\n"
        "        self.checkpoint = checkpoint\n"
        "        self.closed = False\n"
        "    def close(self):\n"
        "        self.closed = True\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Only the external service is substituted; use the real connector and owner.
    connector = importlib.import_module(
        "vllm.distributed.engram_share.uniengram_mapped"
    )
    reader = connector.get_reader(str(tmp_path), full_decode_graph=True)
    assert connector.get_reader() is reader
    assert reader.checkpoint == str(tmp_path.resolve())
    connector.close_reader()
    assert reader.closed
