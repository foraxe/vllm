#!/usr/bin/env python3
"""Exercise the real gather provider lookup against checkpoint-prefix rows.

The synthetic owner uses actual checkpoint bytes. Native setup is stubbed, so
this is a component/data-path test rather than the full provider lifecycle.
"""
import argparse
import base64
import hashlib
import importlib
import json
import os
import statistics
import sys
import threading
import time
import types
from pathlib import Path

import torch
from checkpoint_prefix import read_prefix
from provider_imports import setup

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--owner-ready', type=Path, required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--device', default='mlx5_bond_0')
p.add_argument('--gid', type=int, default=3)
p.add_argument('--rows', type=int, default=1048576)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--native-library', type=Path)
p.add_argument('--native-port', type=int, default=39520)
p.add_argument('--native-qps', type=int, choices=(1, 16), default=1)
p.add_argument('--native-group-base', type=int, default=39600)
p.add_argument('--threshold', type=int, default=1)
p.add_argument('--reader-id', type=int, default=0)
p.add_argument('--pattern', choices=('paired', 'unique'), default='paired')
p.add_argument('--samples', type=int, default=20)
a = p.parse_args()
runtime_module = setup(a.source)
provider = importlib.import_module('vllm.distributed.engram_share.rdma.gather_provider')
native = importlib.import_module('vllm.distributed.engram_share.rdma.provider')
kernels = importlib.import_module('vllm.distributed.engram_share.rdma.gather_kernels')
transport = importlib.import_module('vllm.distributed.engram_share.rdma.gather_transport')

torch.set_num_threads(1)
torch.cuda.set_device(0)
tables = [read_prefix(a.model, layer, a.rows) for layer in (1, 14)]
hashes = [hashlib.sha256(memoryview(t.numpy()).cast('B')).hexdigest() for t in tables]
metadata = json.loads(a.owner_ready.read_text())
if hashes != metadata['prefix_sha256']:
    raise ValueError('reader and owner checkpoint prefix differ')
print('CHECKPOINT_PREFIX_HASH_MATCH', flush=True)
max_queries = 32768
arena = torch.empty(max_queries * 264 + 128, dtype=torch.uint8, device='cuda')
lib = transport.load_library(a.library)
handle = lib.gg_reader_create(0, a.device.encode(), a.gid, 16, arena.data_ptr(), arena.numel(), max_queries)
if not handle:
    transport.check(lib, -1)
transport.check(lib, lib.gg_reader_connect(handle, a.host.encode(), a.port))
reader = object.__new__(provider.GatherReader)
reader.lib, reader.handle, reader.native = lib, handle, None
reader.native_group = None
reader.checkpoint_path = a.model
reader.runtime = types.SimpleNamespace(arena=arena, p_off=64, max_queries=max_queries,
                                       metadata={'lengths': [a.rows * 264, a.rows * 264]})
reader.stats = torch.zeros((16, 8), dtype=torch.uint64, device='cuda')
reader.workspaces = [kernels.workspace(max_queries, 'cuda') for _ in range(2)]
reader.threshold, reader._last_stream, reader._local = a.threshold, None, threading.local()
reader._submission_stream = None
reader.prepared = reader.closed = False
native_context = None
if a.native_library:
    native_context = runtime_module.Context(a.native_library, 0, a.native_port)
    native_context.register([arena], host=False)
    native_context.connect_raw(base64.b64decode(metadata['native_md'], validate=True),
                               metadata['addresses'], metadata['lengths'], metadata['owner_gpu'])
    native_runtime = object.__new__(native.ReaderRuntime)
    native_runtime.context, native_runtime.arena = native_context, arena
    native_runtime.closed, native_runtime.max_queries = False, max_queries
    native_runtime.p_off, native_runtime.dump = 64, max_queries * 264 + 64
    native_runtime.stats = torch.zeros((8,), dtype=torch.uint64, device='cuda')
    native_runtime.metadata = metadata
    reader.runtime = native_runtime
    native_reader = object.__new__(native.VllmRemoteReader)
    native_reader.runtime, native_reader.consumed = native_runtime, torch.cuda.Event()
    native_reader.inflight, native_reader.last_stream, native_reader.captured_stream = False, None, None
    reader.native = native_reader
    if a.native_qps == 16:
        group = importlib.import_module('vllm.distributed.engram_share.rdma.native_group')
        reader.native_group = group.NativeReadGroup(native_runtime, a.native_library, a.native_group_base)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())

def reference(layer, ids, lo, hi):
    flat = ids.view(-1).to(torch.int64)
    owned = (flat >= lo) & (flat < hi)
    safe = torch.where(owned, flat, 0)
    rows = tables[layer][safe]
    values = rows[:, :256].contiguous().view(torch.float8_e4m3fn).float()
    scale = (rows[:, 256:].contiguous().to(torch.int32) << 23).view(torch.float32)
    out = (values * scale.repeat_interleave(32, dim=1)).to(torch.bfloat16)
    out[~owned] = 0
    return out.view(ids.shape[0], 2, 256)

def table_for(layer):
    return types.SimpleNamespace(dim=256, block_size=32, part_n_hash_cols=2,
        head_start=0, vocab_start_idx=17, vocab_end_idx=a.rows-17,
        num_embeddings=a.rows, rdma_layer_id=(1, 14)[layer], rdma_full_decode_graph=True)


def assert_bf16_bits(actual, expected):
    actual_bits = actual.cpu().contiguous().view(torch.int16)
    expected_bits = expected.contiguous().view(torch.int16)
    if not torch.equal(actual_bits, expected_bits):
        raise AssertionError(f'BF16 bit mismatch count={int((actual_bits != expected_bits).sum())}')

results = []
try:
    for count in (64, 1024, 4096, 32768):
        generator = torch.Generator().manual_seed(20261118 + count)
        if a.pattern == 'unique':
            cpu_ids = ((torch.arange(count, dtype=torch.int64) * 104729 + 12345) % a.rows).view(-1, 2)
        else:
            cpu_ids = torch.randint(0, a.rows, (count // 2, 2), generator=generator, dtype=torch.int64)
            cpu_ids.view(-1)[::4] = cpu_ids.view(-1)[1::4]
        flat = cpu_ids.view(-1); flat[1] = -1; flat[2] = a.rows + 3
        ids = cpu_ids.to('cuda')
        output = torch.empty((count // 2, 2, 256), dtype=torch.bfloat16, device='cuda')
        table = table_for(0); expected = reference(0, cpu_ids, table.vocab_start_idx, table.vocab_end_idx)
        stream.wait_stream(torch.cuda.current_stream())
        for _ in range(5):
            with reader.execution_scope(), torch.cuda.stream(stream):
                reader.lookup(table, ids, output)
        torch.cuda.synchronize()
        assert_bf16_bits(output, expected)
        samples=[]
        for _ in range(a.samples):
            begin=time.perf_counter_ns(); cpu_begin=time.process_time_ns(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            with reader.execution_scope(), torch.cuda.stream(stream):
                start.record(); reader.lookup(table, ids, output); end.record(); end.synchronize()
            samples.append({'event_us':start.elapsed_time(end)*1000, 'wall_us':(time.perf_counter_ns()-begin)/1000,
                            'reader_process_cpu_us':(time.process_time_ns()-cpu_begin)/1000})
        assert_bf16_bits(output, expected)
        results.append({'rows':count, 'correct':True, 'event_median_us':statistics.median(s['event_us'] for s in samples),
                        'wall_median_us':statistics.median(s['wall_us'] for s in samples),
                        'reader_process_cpu_median_us':statistics.median(s['reader_process_cpu_us'] for s in samples), 'samples':samples})
        print('LOOKUP_PASS', count, flush=True)
    count=1024; table=table_for(1)
    cpu_ids=torch.arange(count,dtype=torch.int64).view(-1,2)+100
    flat=cpu_ids.view(-1); flat[::4]=flat[1::4]; flat[1]=-1; flat[2]=a.rows+3
    ids=cpu_ids.to('cuda'); output=torch.empty((count//2,2,264),dtype=torch.uint8,device='cuda')
    owned=(flat>=table.vocab_start_idx)&(flat<table.vocab_end_idx)
    raw=tables[1][torch.where(owned,flat,0)].clone();raw[~owned]=0
    stream.wait_stream(torch.cuda.current_stream())
    with reader.execution_scope(), torch.cuda.stream(stream):
        reader.lookup(table,ids,output)
    torch.cuda.synchronize()
    assert torch.equal(output.cpu(),raw.view(count//2,2,264))
    print('PACKED_LAYER14_PASS',flush=True)
    # Exercise both layers with different sizes/routes inside one outer graph.
    counts=(1024,32768); graph_tables=[table_for(0),table_for(1)]
    cpu_inputs=[((torch.arange(n,dtype=torch.int64)*104729+100)%a.rows).view(-1,2) for n in counts]
    graph_ids=[x.to('cuda') for x in cpu_inputs]
    graph_outputs=[torch.empty((n//2,2,256),dtype=torch.bfloat16,device='cuda') for n in counts]
    stream.wait_stream(torch.cuda.current_stream())
    with reader.execution_scope():
        with torch.cuda.stream(stream):
            for table, ids, output in zip(graph_tables,graph_ids,graph_outputs):
                reader.lookup(table,ids,output)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with reader.execution_scope():
        with torch.cuda.graph(graph,stream=stream):
            for table, ids, output in zip(graph_tables,graph_ids,graph_outputs):
                reader.lookup(table,ids,output)
    for shift in (1,37):
        changed=[(x+shift)%a.rows for x in cpu_inputs]
        with reader.execution_scope():
            with torch.cuda.stream(stream):
                for ids, current in zip(graph_ids,changed): ids.copy_(current.to('cuda'))
                graph.replay()
            stream.synchronize()
            for layer,(table,current,output) in enumerate(zip(graph_tables,changed,graph_outputs)):
                assert_bf16_bits(output,reference(layer,current,table.vocab_start_idx,table.vocab_end_idx))
    print('PROVIDER_OUTER_GRAPH_PASS',flush=True)
    reader.prepare_close(); del graph
    transport.check(lib,lib.gg_reader_close(handle));reader.handle=None
    if reader.native_group is not None:
        reader.native_group.close()
    if native_context is not None:
        native_context.close()
    a.out.write_text(json.dumps({'source':'WIP provider; checkpoint owner lifecycle stubbed',
        'native_enabled':bool(a.native_library),'native_qps':a.native_qps,'threshold':a.threshold,'pattern':a.pattern,'rows':a.rows,
        'prefix_sha256':hashes,'results':results,'packed_layer14':True,'outer_graph':True},indent=2)+'\n')
    (a.owner_ready.parent / f'reader-{a.reader_id}.done').write_text('{"ok":true}\n')
    print('PROVIDER_CLOSE_PASS',flush=True)
except BaseException:
    # Preserve graph, arena and any registered resources until process exit.
    print('PROVIDER_FAILED_RESOURCES_HELD_UNTIL_EXIT',flush=True)
    raise
