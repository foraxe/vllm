#!/usr/bin/env python3
"""Two-process candidate transport probe; writes only synthetic rows."""
import argparse
import base64
import ctypes as C
import hashlib
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--role', choices=('owner', 'reader'), required=True)
p.add_argument('--library', required=True)
p.add_argument('--device', required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--gid', type=int, default=3)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--fault', action='store_true')
p.add_argument('--readers', type=int, default=1)
p.add_argument('--reader-id', type=int, default=0)
p.add_argument('--model', type=Path)
p.add_argument('--rows', type=int, default=64)
p.add_argument('--source', type=Path)
p.add_argument('--native-library', type=Path)
p.add_argument('--native-port', type=int, default=39500)
a = p.parse_args()
lib = C.CDLL(a.library)
u64, u32, ptr = C.c_uint64, C.c_uint, C.c_void_p
sig = {
    'gg_owner_create': ([C.c_int, C.c_char_p, C.c_int, u32, C.POINTER(u64), C.POINTER(u64), u32, u32], ptr),
    'gg_owner_close': ([ptr], C.c_int),
    'gg_reader_create': ([C.c_int, C.c_char_p, C.c_int, u32, u64, u64, u32], ptr),
    'gg_reader_connect': ([ptr, C.c_char_p, u32], C.c_int),
    'gg_fetch': ([ptr, u64, u32, u64, u32, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64], C.c_int),
    'gg_reader_begin_execution': ([ptr], C.c_int),
    'gg_reader_begin_execution_wait': ([ptr, u32], C.c_int),
    'gg_reader_end_execution': ([ptr, u64], C.c_int),
    'gg_reader_prepare_close': ([ptr], C.c_int),
    'gg_reader_close': ([ptr], C.c_int),
}
lib.gg_last_error.restype = C.c_char_p
for name, (args, ret) in sig.items():
    f = getattr(lib, name); f.argtypes = args; f.restype = ret


def check(result):
    if result:
        raise RuntimeError((lib.gg_last_error() or b'unknown error').decode())


def expected(ids, layer):
    r = torch.tensor(ids, dtype=torch.int64).view(-1, 1)
    c = torch.arange(264, dtype=torch.int64).view(1, -1)
    return ((r * 7 + c * 11 + layer * 13) % 251).to(torch.uint8)


torch.cuda.set_device(0)
a.out.mkdir(parents=True, exist_ok=True)
if a.role == 'owner':
    def cpu_snapshot(_signum, _frame):
        with (a.out / 'owner-cpu.jsonl').open('a') as f:
            f.write(json.dumps({'wall_ns': time.monotonic_ns(), 'cpu_ns': time.process_time_ns()}) + '\n')
    signal.signal(signal.SIGUSR1, cpu_snapshot)
    if a.model:
        from checkpoint_prefix import read_prefix
        tables = [read_prefix(a.model, layer, a.rows).pin_memory() for layer in (1, 14)]
    else:
        tables = [expected(list(range(64)), i).pin_memory() for i in range(2)]
    addrs = (u64 * 2)(*(t.data_ptr() for t in tables))
    sizes = (u64 * 2)(*(t.numel() for t in tables))
    native_context = None
    native_metadata = {}
    if a.native_library:
        if a.source is None:
            raise ValueError('--source is required for native registration')
        from provider_imports import setup
        runtime = setup(a.source)
        native_context = runtime.Context(a.native_library, 0, a.native_port)
        native_context.register(tables, host=True)
        native_metadata = {
            'native_md': base64.b64encode(native_context.metadata()).decode(),
            'addresses': list(native_context.addresses),
            'lengths': list(native_context.lengths),
            'owner_gpu': 0,
        }
    handles = []
    for index in range(a.readers):
        handle = lib.gg_owner_create(0, a.device.encode(), a.gid, 16, addrs, sizes, 2, a.port + index)
        if not handle:
            raise RuntimeError((lib.gg_last_error() or b'owner create').decode())
        handles.append(handle)
    hashes = [hashlib.sha256(memoryview(t.numpy()).cast('B')).hexdigest() for t in tables]
    ready = a.out / 'owner.ready'
    ready.write_text(json.dumps({'prefix_sha256': hashes, **native_metadata}) + '\n')
    ready.chmod(0o600)
    print('OWNER_READY', flush=True)
    deadline = time.monotonic() + 300
    finished = lambda: all((a.out / f'reader-{i}.done').exists() for i in range(a.readers))
    while not finished() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not finished():
        raise TimeoutError('reader did not finish; keep process resources until exit')
    for handle in handles:
        rc = lib.gg_owner_close(handle)
        if a.fault:
            assert rc != 0, 'faulty owner unexpectedly released resources'
        else:
            check(rc)
    if native_context is not None and not a.fault:
        native_context.close()
    print('POISONED_OWNER_RETAINED_PASS' if a.fault else 'OWNER_CLOSE_PASS', flush=True)
else:
    handle = None
    arena = torch.empty(64 * 264, dtype=torch.uint8, device='cuda')
    ids = torch.tensor([7, 3, 7, 1, 12, 0, 63, 5], dtype=torch.int64, device='cuda')
    stats = torch.zeros((16, 8), dtype=torch.uint64, device='cuda')
    out = torch.empty((ids.numel(), 264), dtype=torch.uint8, device='cuda')
    handle = lib.gg_reader_create(0, a.device.encode(), a.gid, 16, arena.data_ptr(), arena.numel(), ids.numel())
    if not handle:
        raise RuntimeError((lib.gg_last_error() or b'reader create').decode())
    check(lib.gg_reader_connect(handle, a.host.encode(), a.port))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    if a.fault:
        check(lib.gg_reader_begin_execution(handle))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            check(lib.gg_fetch(handle, ids.data_ptr(), ids.numel(), 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, stats.data_ptr(), stream.cuda_stream))
            out.copy_(arena[:ids.numel() * 264].view_as(out))
        check(lib.gg_reader_end_execution(handle, stream.cuda_stream))
        check(lib.gg_reader_begin_execution_wait(handle, 10000))
        with torch.cuda.stream(stream):
            graph.replay()
            check(lib.gg_reader_end_execution(handle, stream.cuda_stream))
        stream.synchronize()
        assert int(stats.view(torch.int64)[:, 0].max()) != 0
        assert lib.gg_reader_begin_execution(handle) != 0
        assert lib.gg_reader_prepare_close(handle) != 0
        assert lib.gg_reader_close(handle) != 0
        (a.out / f'reader-{a.reader_id}.done').write_text(json.dumps({'fault_retained': True}) + '\n')
        print('GRAPH_FAULT_AND_RETENTION_PASS', flush=True)
        # Keep graph and every associated tensor alive until process exit.
        raise SystemExit(0)
    check(lib.gg_reader_begin_execution(handle))
    assert lib.gg_reader_begin_execution(handle) != 0
    assert lib.gg_reader_begin_execution_wait(handle, 20) != 0
    assert lib.gg_reader_prepare_close(handle) != 0
    with torch.cuda.stream(stream):
        check(lib.gg_fetch(handle, ids.data_ptr(), ids.numel(), 0, 0, 0, 64, 64, 0, 0, 0, 0, 0, stats.data_ptr(), stream.cuda_stream))
        out.copy_(arena[:ids.numel() * 264].view_as(out))
        torch.cuda._sleep(100_000_000)
        check(lib.gg_reader_end_execution(handle, stream.cuda_stream))
    # Fail closed while a consumer is pending; if scheduling completed quickly,
    # accepting the next lease is also correct and is handled below.
    rc = lib.gg_reader_begin_execution(handle)
    if rc:
        check(lib.gg_reader_begin_execution_wait(handle, 10000))
    assert torch.equal(out.cpu(), expected(ids.cpu().tolist(), 0))
    print('EAGER_AND_EXCLUSIVE_LEASE_PASS', flush=True)
    # Begin is held. Capture direct operations into an outer graph, then release
    # the capture lease outside capture and acquire again before each replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        check(lib.gg_fetch(handle, ids.data_ptr(), ids.numel(), 0, 1, 0, 64, 64, 0, 0, 0, 0, 0, stats.data_ptr(), stream.cuda_stream))
        out.copy_(arena[:ids.numel() * 264].view_as(out))
    check(lib.gg_reader_end_execution(handle, stream.cuda_stream))
    for changed in ([3, 4, 5, 6, 7, 8, 9, 10], [9, 9, 1, 0, 63, 4, 2, 2]):
        check(lib.gg_reader_begin_execution_wait(handle, 10000))
        with torch.cuda.stream(stream):
            ids.copy_(torch.tensor(changed, dtype=torch.int64, pin_memory=True), non_blocking=True)
            graph.replay()
            check(lib.gg_reader_end_execution(handle, stream.cuda_stream))
        stream.synchronize()
        assert int(stats.view(torch.int64)[:, 0].max()) == 0
        assert torch.equal(out.cpu(), expected(changed, 1))
    print('OUTER_GRAPH_CHANGED_IDS_PASS', flush=True)
    gate = threading.Barrier(2)

    def replay_on_stream(worker):
        torch.cuda.set_device(0)
        own_stream = torch.cuda.Stream()
        gate.wait()
        for step in range(4):
            changed = [(worker * 17 + step * 7 + i) % 64 for i in range(ids.numel())]
            check(lib.gg_reader_begin_execution_wait(handle, 10000))
            with torch.cuda.stream(own_stream):
                ids.copy_(torch.tensor(changed, dtype=torch.int64, pin_memory=True), non_blocking=True)
                graph.replay()
            own_stream.synchronize()
            assert torch.equal(out.cpu(), expected(changed, 1))
            check(lib.gg_reader_end_execution(handle, own_stream.cuda_stream))
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(replay_on_stream, range(2)))
    print('TWO_STREAM_SERIALIZED_GRAPH_PASS', flush=True)
    check(lib.gg_reader_prepare_close(handle))
    # After prepare succeeds no new API submissions are accepted; destroy the
    # captured graph while registered storage is still held, then close.
    assert lib.gg_reader_begin_execution(handle) != 0
    del graph
    check(lib.gg_reader_close(handle))
    (a.out / f'reader-{a.reader_id}.done').write_text(json.dumps({'ok': True}) + '\n')
    print('TWO_PHASE_CLOSE_PASS', flush=True)
