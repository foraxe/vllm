# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from vllm.distributed.engram_share import execution
from vllm.distributed.engram_share.rdma import gather_provider, native_group
from vllm.distributed.engram_share.rdma.gather_transport import UnsafeRetirementError


def test_runner_scope_wraps_inputs_and_propagates_errors(monkeypatch):
    events = []

    @contextmanager
    def scope():
        events.append("begin")
        try:
            yield
        finally:
            events.append("end")

    monkeypatch.setattr(
        execution, "get_provider", lambda _: SimpleNamespace(execution_scope=scope)
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(engram_config=SimpleNamespace(rdma_provider="test"))
    )

    @execution.with_provider_execution
    def forward(self):
        events.append("write_inputs_and_launch")
        raise ValueError("example")

    with pytest.raises(ValueError, match="example"):
        forward(runner)
    assert events == ["begin", "write_inputs_and_launch", "end"]


class FakeLibrary:
    def __init__(self):
        self.calls = []

    def gg_reader_begin_execution_wait(self, handle, timeout):
        self.calls.append(("begin", handle, timeout))
        return 0

    def gg_reader_end_execution(self, handle, stream):
        self.calls.append(("end", handle, stream))
        return 0

    def gg_reader_abort_execution(self, handle):
        self.calls.append(("abort", handle))
        return 0


def make_reader():
    reader = object.__new__(gather_provider.GatherReader)
    reader.lib, reader.handle = FakeLibrary(), 42
    reader.closed = reader.prepared = False
    reader._local = threading.local()
    reader._last_stream = None
    return reader


def test_nested_execution_uses_one_lease(monkeypatch):
    reader = make_reader()
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(gather_provider.torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(
        gather_provider.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    with reader.execution_scope(), reader.execution_scope():
        assert reader._local.depth == 2
    assert reader.lib.calls == [("begin", 42, 10000), ("end", 42, 7)]


def test_failed_execution_quarantines_reader(monkeypatch):
    reader = make_reader()
    with pytest.raises(ValueError, match="failed"), reader.execution_scope():
        raise ValueError("failed")
    assert reader.lib.calls == [("begin", 42, 10000), ("abort", 42)]
    assert reader in gather_provider._quarantined
    gather_provider._quarantined.remove(reader)


def test_native_group_partial_setup_failure_retains_shared_context(monkeypatch):
    class Function:
        def __call__(self, *args):
            return 0

    lib = SimpleNamespace(
        engram_read_group_create=Function(),
        engram_read_group_fetch_compacted=Function(),
        engram_read_group_destroy=Function(),
        engram_read_group_pipeline=Function(),
    )
    first = SimpleNamespace(library=lib, gpu=0)
    runtime = SimpleNamespace(
        context=first,
        arena=object(),
        metadata={"native_md": "", "addresses": [1], "lengths": [1], "owner_gpu": 0},
    )

    class FailedContext:
        def __init__(self, *args):
            pass

        def register(self, *args, **kwargs):
            raise RuntimeError("registration failed")

        def close(self):
            raise RuntimeError("retirement failed")

    monkeypatch.setattr(native_group, "Context", FailedContext)
    monkeypatch.setattr(native_group.torch, "zeros", lambda *a, **k: object())
    with pytest.raises(UnsafeRetirementError, match="could not be retired"):
        native_group.NativeReadGroup(runtime, "unused", 40000)
    group = native_group._quarantined_groups.pop()
    assert group.contexts[0] is first


def test_unsafe_native_setup_keeps_reader_and_owner_lease(tmp_path, monkeypatch):
    class Arena:
        def data_ptr(self):
            return 100

        def numel(self):
            return 1000

    class Native:
        closed = False

        def __init__(self, _):
            self.runtime = SimpleNamespace(
                metadata={
                    "gather": {"protocol": "gather_graph_v1", "host": "h", "port": 1}
                },
                arena=Arena(),
                max_queries=2,
                close=lambda: None,
            )

        def close(self):
            self.closed = True

    class Library:
        def gg_reader_create(self, *args):
            return 7

        def gg_reader_connect(self, *args):
            return 0

    def fail(*args):
        raise UnsafeRetirementError("could not be retired")

    monkeypatch.setattr(gather_provider, "VllmRemoteReader", Native)
    monkeypatch.setattr(gather_provider, "load_library", lambda _: Library())
    monkeypatch.setattr(gather_provider.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(gather_provider.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(gather_provider.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(gather_provider.gather_kernels, "workspace", lambda *a: {})
    monkeypatch.setattr(native_group, "NativeReadGroup", fail)
    for key, value in {
        "ENGRAM_GATHER_LIBRARY": "unused",
        "ENGRAM_GATHER_DEVICE": "rdma0",
        "ENGRAM_GATHER_NATIVE_QPS": "16",
        "ENGRAM_GATHER_NATIVE_PORT_BASE": "40000",
        "ENGRAM_SHARED_LIBRARY": "unused",
        "ENGRAM_SHARED_READER_PORT_BASE": "39000",
    }.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(UnsafeRetirementError):
        gather_provider.GatherReader(tmp_path)
    reader = gather_provider._quarantined.pop()
    assert not reader.native.closed and reader.handle == 7


def test_group_port_ranges_are_disjoint_across_ranks():
    native_group.validate_port_layout(27000, 30000, 4)
    with pytest.raises(ValueError, match="overlap"):
        native_group.validate_port_layout(30000, 30000, 4)
    with pytest.raises(ValueError, match="invalid"):
        native_group.validate_port_layout(27000, 65520, 4)
