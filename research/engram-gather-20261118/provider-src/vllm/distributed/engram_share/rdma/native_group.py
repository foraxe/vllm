# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in grouped native READs for the gathered-reader comparison."""

import base64
import ctypes as c

import torch

from vllm.distributed.engram_share.rdma.runtime import Context

_quarantined_groups = []


class NativeReadGroup:
    """Borrow the native reader's first context and own 15 additional QPs."""

    def __init__(self, runtime, library, port_base):
        first = runtime.context
        self.contexts = [first]
        self.handle = None
        self.lib = first.library
        self.stats = [
            torch.zeros((16, 8), dtype=torch.uint64, device="cuda") for _ in range(2)
        ]
        lib = self.lib
        lib.engram_read_group_create.argtypes = [c.POINTER(c.c_void_p), c.c_uint32]
        lib.engram_read_group_create.restype = c.c_void_p
        lib.engram_read_group_fetch_compacted.argtypes = [
            c.c_void_p,
            c.c_uint64,
            c.c_uint32,
            c.c_size_t,
            c.c_size_t,
            c.c_uint32,
            c.c_uint32,
            c.c_uint64,
            c.c_uint64,
            c.c_uint64,
            c.c_uint64,
            c.c_int,
            c.c_uint64,
            c.c_uint64,
            c.c_uint64,
        ]
        lib.engram_read_group_destroy.argtypes = [c.c_void_p]
        lib.engram_read_group_pipeline.argtypes = [c.c_void_p, c.c_uint32]
        try:
            md = runtime.metadata
            for index in range(1, 16):
                context = Context(library, first.gpu, port_base + index)
                self.contexts.append(context)
                context.register([runtime.arena], host=False)
                context.connect_raw(
                    base64.b64decode(md["native_md"], validate=True),
                    md["addresses"],
                    md["lengths"],
                    md["owner_gpu"],
                )
            handles = (c.c_void_p * 16)(*(context.handle for context in self.contexts))
            self.handle = lib.engram_read_group_create(handles, 16)
            if not self.handle:
                raise RuntimeError(lib.engram_last_error().decode())
            first.check(lib.engram_read_group_pipeline(self.handle, 2))
        except BaseException:
            try:
                self.close()
            except BaseException:
                _quarantined_groups.append(self)
            raise

    def fetch(self, ids, layer, payload, dump, lo, hi, workspace):
        stream = torch.cuda.current_stream(self.contexts[0].gpu)
        stats = self.stats[layer]
        self.contexts[0].check(
            self.lib.engram_read_group_fetch_compacted(
                self.handle,
                ids.data_ptr(),
                ids.numel(),
                payload,
                dump,
                layer,
                512,
                stats.data_ptr(),
                stream.cuda_stream,
                lo,
                hi,
                0,
                workspace["reps"].data_ptr(),
                workspace["work"].data_ptr(),
                workspace["scratch"][-1:].data_ptr(),
            )
        )
        ids.record_stream(stream)
        torch._assert_async(
            torch.all(stats.view(torch.int64)[:, 0] == 0), "native READ failed"
        )

    def record_consumed(self):
        for context in self.contexts:
            context.record_consumed()

    def close(self):
        try:
            if self.handle:
                self.contexts[0].check(self.lib.engram_read_group_destroy(self.handle))
                self.handle = None
            while len(self.contexts) > 1:
                self.contexts[-1].close()
                self.contexts.pop()
        except BaseException:
            _quarantined_groups.append(self)
            raise
