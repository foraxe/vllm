# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental opt-in gathered-row reader; requires an execution lease."""

import atexit
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from vllm.distributed.engram_share.rdma import gather_kernels
from vllm.distributed.engram_share.rdma.gather_transport import (
    UnsafeRetirementError,
    check,
    load_library,
)
from vllm.distributed.engram_share.rdma.provider import VllmRemoteReader

_quarantined = []
_reader = None


class GatherReader:
    """One reader/arena; graph inputs and both layer lookups share one lease."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        self.native = VllmRemoteReader(checkpoint_path)
        self.runtime = self.native.runtime
        # The gather owner must retire before native runtime releases its lease.
        atexit.unregister(self.runtime.close)
        self.lib = None
        self.handle = None
        self.native_group = None
        self.prepared = False
        self.closed = False
        self._local = threading.local()
        self._last_stream = None
        self._submission_stream = None
        try:
            metadata = self.runtime.metadata.get("gather")
            if not metadata or metadata.get("protocol") != "gather_graph_v1":
                raise ValueError("owner did not provide a gather_graph_v1 endpoint")
            self.lib = load_library(os.environ["ENGRAM_GATHER_LIBRARY"])
            gpu = torch.cuda.current_device()
            self.handle = self.lib.gg_reader_create(
                gpu,
                os.environ["ENGRAM_GATHER_DEVICE"].encode(),
                int(os.environ.get("ENGRAM_GATHER_GID_INDEX", "0")),
                16,
                self.runtime.arena.data_ptr(),
                self.runtime.arena.numel(),
                self.runtime.max_queries,
            )
            if not self.handle:
                check(self.lib, -1)
            check(
                self.lib,
                self.lib.gg_reader_connect(
                    self.handle, metadata["host"].encode(), int(metadata["port"])
                ),
            )
            self.stats = torch.zeros((16, 8), dtype=torch.uint64, device="cuda")
            self.workspaces = [
                gather_kernels.workspace(self.runtime.max_queries, "cuda")
                for _ in range(2)
            ]
            self.threshold = int(os.environ.get("ENGRAM_GATHER_THRESHOLD", "32768"))
            if self.threshold < 1:
                raise ValueError("gather threshold must be positive")
            qps = int(os.environ.get("ENGRAM_GATHER_NATIVE_QPS", "1"))
            if qps not in (1, 16):
                raise ValueError("gather native fallback supports one or sixteen QPs")
            if qps == 16:
                from vllm.distributed.engram_share.rdma.native_group import (
                    NativeReadGroup,
                    validate_port_layout,
                )

                group_base = int(os.environ["ENGRAM_GATHER_NATIVE_PORT_BASE"])
                validate_port_layout(
                    int(os.environ["ENGRAM_SHARED_READER_PORT_BASE"]),
                    group_base,
                    torch.cuda.device_count(),
                )
                base = group_base + gpu * 32
                self.native_group = NativeReadGroup(
                    self.runtime, os.environ["ENGRAM_SHARED_LIBRARY"], base
                )
        except BaseException as exc:
            if isinstance(exc, UnsafeRetirementError):
                _quarantined.append(self)
                raise
            try:
                self.prepare_close()
                self.close()
            except BaseException:
                _quarantined.append(self)
            raise

    @contextmanager
    def execution_scope(self):
        """Enclose every graph launch and all writes to its shared input buffers."""
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            if self.closed or self.prepared:
                raise RuntimeError("gather reader is closing")
            timeout = int(os.environ.get("ENGRAM_GATHER_WAIT_MS", "10000"))
            check(
                self.lib, self.lib.gg_reader_begin_execution_wait(self.handle, timeout)
            )
        self._local.depth = depth + 1
        try:
            yield
        except BaseException:
            if depth == 0:
                self.lib.gg_reader_abort_execution(self.handle)
                _quarantined.append(self)
            raise
        else:
            if depth == 0:
                try:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "gather lease must end outside graph capture"
                        )
                    stream = torch.cuda.current_stream()
                    if (
                        self._last_stream is not None
                        and stream.cuda_stream != self._last_stream.cuda_stream
                    ):
                        stream.wait_stream(self._last_stream)
                    check(
                        self.lib,
                        self.lib.gg_reader_end_execution(
                            self.handle, stream.cuda_stream
                        ),
                    )
                except BaseException:
                    self.lib.gg_reader_abort_execution(self.handle)
                    _quarantined.append(self)
                    raise
        finally:
            self._local.depth = depth

    def lookup(self, table, indices, out, background=False):
        if not getattr(self._local, "depth", 0):
            raise RuntimeError("gather lookup requires an execution lease")
        stream = torch.cuda.current_stream()
        if self._submission_stream is None:
            self._submission_stream = stream.cuda_stream
        elif self._submission_stream != stream.cuda_stream:
            raise RuntimeError("gather reader lookups must use one submission stream")
        count = indices.shape[0] * table.part_n_hash_cols
        if count < self.threshold and (self.native_group is None or count < 4096):
            self.native.lookup(table, indices, out, background)
            self._last_stream = stream
            return
        if count > self.runtime.max_queries:
            raise ValueError("gather scratch capacity exceeded")
        if table.dim != 256 or table.block_size != 32:
            raise ValueError("unsupported remote Engram row format")
        if out.dtype not in (torch.bfloat16, torch.uint8) or out.ndim != 3:
            raise ValueError("expected 3-D BF16 or packed-byte output")
        if out.shape != (
            indices.shape[0],
            table.part_n_hash_cols,
            264 if out.dtype == torch.uint8 else 256,
        ):
            raise ValueError("output shape does not match local Engram heads")
        start, heads = table.head_start, table.part_n_hash_cols
        local = indices[:, start : start + heads]
        if local.shape[1] < heads:
            local = F.pad(local, (0, heads - local.shape[1]), value=-1)
        ids = local.to(dtype=torch.int64).contiguous().view(-1)
        read_ids = torch.where((ids >= 0) & (ids < table.num_embeddings), ids, 0)
        layer = (1, 14).index(table.rdma_layer_id)
        w = self.workspaces[layer]
        gather_kernels.prepare(
            read_ids,
            table.vocab_start_idx,
            table.vocab_end_idx,
            w["scratch"],
            w["reps"][:count],
            w["work"][:count],
        )
        if count >= self.threshold:
            check(
                self.lib,
                self.lib.gg_fetch(
                    self.handle,
                    read_ids.data_ptr(),
                    count,
                    self.runtime.p_off,
                    layer,
                    table.vocab_start_idx,
                    table.vocab_end_idx,
                    self.runtime.metadata["lengths"][layer] // 264,
                    w["reps"].data_ptr(),
                    w["work"].data_ptr(),
                    w["scratch"][-1:].data_ptr(),
                    w["dense"].data_ptr(),
                    w["inverse"].data_ptr(),
                    self.stats.data_ptr(),
                    stream.cuda_stream,
                ),
            )
            torch._assert_async(
                torch.all(self.stats.view(torch.int64)[:, 0] == 0), "gather failed"
            )
            mapping = w["dense"]
        else:
            self.native_group.fetch(
                read_ids,
                layer,
                self.runtime.p_off,
                self.runtime.dump,
                table.vocab_start_idx,
                table.vocab_end_idx,
                w,
            )
            mapping = w["reps"]
        raw = self.runtime.arena[
            self.runtime.p_off : self.runtime.p_off + count * 264
        ].view(count, 264)
        weight, scales = raw[:, :256].view(torch.float8_e4m3fn), raw[:, 256:]
        gather_kernels._scatter_mapped[(count,)](
            weight,
            scales,
            ids,
            mapping,
            out,
            table.vocab_start_idx,
            table.vocab_end_idx,
            weight.stride(0),
            scales.stride(0),
            out.stride(0),
            out.stride(1),
            heads,
            out.dtype == torch.uint8,
            num_warps=4,
        )
        if count >= self.threshold:
            check(
                self.lib, self.lib.gg_record_consumed(self.handle, stream.cuda_stream)
            )
        else:
            self.native_group.record_consumed()
        self._last_stream = stream

    def prepare_close(self):
        try:
            if self.handle and not self.prepared:
                check(self.lib, self.lib.gg_reader_prepare_close(self.handle))
            self.prepared = True
        except BaseException:
            _quarantined.append(self)
            raise

    def close(self):
        if self.closed:
            return
        if not self.prepared:
            raise RuntimeError("prepare_close must succeed before releasing graphs")
        try:
            if self.handle:
                check(self.lib, self.lib.gg_reader_close(self.handle))
                self.handle = None
            if self.native_group is not None:
                self.native_group.close()
            self.native.close()
            self.closed = True
        except BaseException:
            _quarantined.append(self)
            raise


def get_reader(checkpoint_path=None):
    global _reader
    if _reader is None:
        if checkpoint_path is None:
            raise RuntimeError("gather reader must be initialized by the model loader")
        _reader = GatherReader(checkpoint_path)
    elif (
        checkpoint_path is not None
        and Path(checkpoint_path).resolve(strict=True) != _reader.checkpoint_path
    ):
        raise ValueError("gather reader belongs to a different engine checkpoint")
    return _reader


@contextmanager
def execution_scope():
    with get_reader().execution_scope():
        yield


def prepare_close():
    if _reader is not None:
        try:
            _reader.prepare_close()
        except BaseException:
            _quarantined.append(_reader)
            raise


def quarantine_graph_owner(owner):
    # The runner owns the captured graphs and their static input buffers.
    _quarantined.append(owner)


def close_reader():
    global _reader
    if _reader is not None:
        try:
            _reader.close()
        except BaseException:
            _quarantined.append(_reader)
            raise
        _reader = None
