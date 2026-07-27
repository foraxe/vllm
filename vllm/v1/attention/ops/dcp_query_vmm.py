# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owner-local CUDA VMM transport for DCP MLA query-head shards.

Each rank produces only its local FP8 query heads directly into owner-local VMM
storage. After device-side publication, the sparse-attention consumer reads
each owner's peer shard directly. No full-head query tensor is materialized.

The workspace is reusable and CUDA-graph safe. Device sequence counters prevent
an owner from overwriting a generation before every consumer has read it.
Initialization and selected-route failures are fail-closed; intentional large-row or
non-decode collective routing must be selected before invoking this workspace.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.profiler import record_function

from vllm.distributed.device_communicators.cuda_vmm import (
    RankMajorPeerView,
    create_rank_major_peer_view,
)
from vllm.distributed.device_communicators.peer_memory import (
    make_rank_major_tensor_view,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SIGNAL_RESERVE_BYTES = 256
_SIGNAL_BYTES = 2 * 8
_WRITE_SEQ = tl.constexpr(0)
_READ_SEQ = tl.constexpr(1)
_MAX_FENCE_SPINS = 100_000_000
DEFAULT_MAX_ROWS = 128
_logged_consume_rows: set[int] = set()


@triton.jit
def _trap_if_nonzero(value):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_writable_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    my_write_seq = tl.atomic_add(
        peer_flags + my_rank * peer_stride + _WRITE_SEQ,
        0,
        sem="acquire",
        scope="sys",
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride + _READ_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < my_write_seq), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride + _READ_SEQ,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < my_write_seq), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _publish_and_wait_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
    max_spins: tl.constexpr,
):
    epoch = (
        tl.atomic_add(
            peer_flags + my_rank * peer_stride + _WRITE_SEQ,
            1,
            sem="release",
            scope="sys",
        )
        + 1
    )
    peer = tl.arange(0, block_size)
    mask = peer < world_size
    observed = tl.atomic_add(
        peer_flags + peer * peer_stride + _WRITE_SEQ,
        0,
        mask=mask,
        sem="acquire",
        scope="sys",
    )
    pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            peer_flags + peer * peer_stride + _WRITE_SEQ,
            0,
            mask=mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(mask & (observed < epoch), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)


@triton.jit
def _ack_kernel(
    peer_flags,
    peer_stride,
    my_rank: tl.constexpr,
):
    tl.atomic_add(
        peer_flags + my_rank * peer_stride + _READ_SEQ,
        1,
        sem="release",
        scope="sys",
    )


@dataclass
class DcpQueryVmmWorkspace:
    my_rank: int
    world_size: int
    max_rows: int
    local_heads: int
    query_dim: int
    group: ProcessGroup
    device: torch.device
    allocation: RankMajorPeerView
    local_query_shard: torch.Tensor
    peer_query: torch.Tensor
    peer_flags: torch.Tensor
    publishing_rows: int = 0
    consumer_rows: int = 0

    @property
    def total_heads(self) -> int:
        return self.local_heads * self.world_size

    @property
    def physical_bytes_per_rank(self) -> int:
        return self.allocation.bytes_per_rank

    @property
    def payload_bytes_per_rank(self) -> int:
        return self.max_rows * self.local_heads * self.query_dim

    def _validate_live(self) -> None:
        if self.allocation.closed:
            raise RuntimeError("DCP query VMM workspace is closed.")
        current_device = torch.accelerator.current_device_index()
        if current_device != self.device.index:
            raise RuntimeError(
                "DCP query VMM current device changed after initialization: "
                f"workspace={self.device}, current=cuda:{current_device}."
            )

    def begin_publish(self, rows: int) -> torch.Tensor:
        """Wait for reuse safety and return owner-local producer storage."""
        self._validate_live()
        if self.publishing_rows:
            raise RuntimeError("DCP query VMM publication is already in progress.")
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                "DCP query VMM producer row bound violated: "
                f"max_rows={self.max_rows}, requested={rows}."
            )
        with record_function("dcp.query_vmm.wait_reuse"):
            _wait_writable_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )
        self.publishing_rows = rows
        return self.local_query_shard[:rows]

    def acquire_peer_query(self, rows: int) -> torch.Tensor:
        """Publish the owner shard and expose one strided peer query view."""
        self._validate_live()
        if not self.publishing_rows:
            raise RuntimeError(
                "DCP query VMM acquire requires begin_publish on this call."
            )
        if self.consumer_rows:
            raise RuntimeError("DCP query VMM consumer read is already in progress.")
        if rows <= 0 or rows > self.max_rows:
            raise RuntimeError(
                "DCP query VMM consumer row bound violated: "
                f"max_rows={self.max_rows}, requested={rows}."
            )
        if rows > self.publishing_rows:
            raise RuntimeError(
                "DCP query VMM consumer rows exceed the producer rows: "
                f"producer={self.publishing_rows}, consumer={rows}."
            )
        self.publishing_rows = 0
        self.consumer_rows = rows
        if rows not in _logged_consume_rows:
            _logged_consume_rows.add(rows)
            logger.info(
                "Executing owner-local CUDA VMM DCP query direct-consume "
                "for decode rows=%d.",
                rows,
            )

        with record_function("dcp.query_vmm.publish_owner_and_acquire_peers"):
            _publish_and_wait_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                max_spins=_MAX_FENCE_SPINS,
            )

        return self.peer_query[:rows]

    def acknowledge(self) -> None:
        """Release the owner shards after every direct consumer has finished."""
        self._validate_live()
        if not self.consumer_rows:
            raise RuntimeError("DCP query VMM acknowledge has no active consumer read.")
        self.consumer_rows = 0
        with record_function("dcp.query_vmm.ack"):
            _ack_kernel[(1,)](
                self.peer_flags,
                self.peer_flags.stride(0),
                my_rank=self.my_rank,
            )

    def close(self) -> None:
        if self.allocation.closed:
            return
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        self.peer_flags = None
        self.peer_query = None
        self.local_query_shard = None
        self.allocation.close()


def create_dcp_query_vmm_workspace_for_group(
    max_rows: int,
    local_heads: int,
    query_dim: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpQueryVmmWorkspace:
    """Collectively create one owner-local FP8 query workspace."""
    world_size = group.size()
    rank = group.rank()
    if world_size <= 1:
        raise RuntimeError("DCP query VMM requires dcp_world_size > 1.")
    if max_rows <= 0 or local_heads <= 0 or query_dim <= 0:
        raise ValueError(
            "DCP query VMM dimensions must be positive; "
            f"got max_rows={max_rows}, local_heads={local_heads}, "
            f"query_dim={query_dim}."
        )

    query_shard_bytes = max_rows * local_heads * query_dim
    allocation = create_rank_major_peer_view(
        (query_shard_bytes + _SIGNAL_RESERVE_BYTES,),
        dtype=torch.uint8,
        group=group,
        require_native_atomics=True,
        device=device,
    )
    assert allocation.local_view is not None
    assert allocation.global_view is not None
    allocation.local_view.zero_()
    torch.accelerator.synchronize()
    dist.barrier(group=group)

    if allocation.bytes_per_rank % local_heads:
        raise RuntimeError(
            "DCP query VMM data segment cannot be split into equal head planes: "
            f"bytes_per_rank={allocation.bytes_per_rank}, "
            f"local_heads={local_heads}."
        )
    head_stride = allocation.bytes_per_rank // local_heads
    if max_rows * query_dim > head_stride:
        raise RuntimeError(
            "DCP query VMM head plane is too small for the configured row bound: "
            f"head_stride={head_stride}, required={max_rows * query_dim}."
        )
    signal_offset = allocation.bytes_per_rank - _SIGNAL_BYTES
    query_span = (local_heads - 1) * head_stride + max_rows * query_dim
    if query_span > signal_offset:
        raise RuntimeError(
            "DCP query VMM allocation has no safe signal tail after the "
            f"strided query: query_span={query_span}, "
            f"signal_offset={signal_offset}."
        )
    local_flags = allocation.local_view[
        signal_offset : signal_offset + _SIGNAL_BYTES
    ].view(torch.int64)
    peer_flags = make_rank_major_tensor_view(allocation, local_flags)
    local_query_shard = torch.as_strided(
        allocation.local_view.view(torch.float8_e4m3fn),
        size=(max_rows, local_heads, query_dim),
        stride=(query_dim, head_stride, 1),
    )
    peer_query = (
        make_rank_major_tensor_view(allocation, local_query_shard)
        .permute(1, 0, 2, 3)
        .flatten(1, 2)
    )
    expected_peer_stride = (query_dim, head_stride, 1)
    if (
        peer_query.data_ptr() != allocation.global_view.data_ptr()
        or peer_query.stride() != expected_peer_stride
    ):
        raise RuntimeError(
            "DCP query VMM peer view was materialized or has an unsupported "
            f"layout: data_ptr={peer_query.data_ptr()}, "
            f"global_ptr={allocation.global_view.data_ptr()}, "
            f"strides={peer_query.stride()}, expected={expected_peer_stride}."
        )
    return DcpQueryVmmWorkspace(
        my_rank=rank,
        world_size=world_size,
        max_rows=max_rows,
        local_heads=local_heads,
        query_dim=query_dim,
        group=group,
        device=device,
        allocation=allocation,
        local_query_shard=local_query_shard,
        peer_query=peer_query,
        peer_flags=peer_flags,
    )


_workspace: DcpQueryVmmWorkspace | None = None
_workspace_failed = False


def get_dcp_query_vmm_workspace(
    max_rows: int,
    local_heads: int,
    query_dim: int,
    group: ProcessGroup,
    device: torch.device,
) -> DcpQueryVmmWorkspace:
    """Create or fetch the singleton query workspace, refusing fallback."""
    global _workspace, _workspace_failed
    world_size = group.size()
    if _workspace_failed:
        raise RuntimeError("DCP query VMM workspace is unavailable.")
    if _workspace is not None:
        actual = (
            _workspace.max_rows,
            _workspace.local_heads,
            _workspace.query_dim,
            _workspace.world_size,
        )
        requested = (max_rows, local_heads, query_dim, world_size)
        requested_device = torch.device(device)
        if requested_device.index is None:
            requested_device = torch.device(
                f"cuda:{torch.accelerator.current_device_index()}"
            )
        if (
            actual != requested
            or _workspace.group is not group
            or _workspace.my_rank != group.rank()
            or _workspace.device != requested_device
        ):
            raise RuntimeError(
                "DCP query VMM workspace identity changed after initialization: "
                f"workspace_geometry={actual}, request_geometry={requested}, "
                f"workspace_rank={_workspace.my_rank}, request_rank={group.rank()}, "
                f"workspace_device={_workspace.device}, "
                f"request_device={requested_device}, "
                f"same_group={_workspace.group is group}."
            )
        return _workspace

    try:
        _workspace = create_dcp_query_vmm_workspace_for_group(
            max_rows,
            local_heads,
            query_dim,
            group,
            device,
        )
    except Exception as exc:
        _workspace_failed = True
        raise RuntimeError(
            "DCP query VMM workspace initialization failed; refusing to "
            "fall back to an explicit collective path."
        ) from exc
    logger.info_once(
        "Using owner-local CUDA VMM DCP query direct-consume "
        "(max_rows=%d, local_heads=%d, total_heads=%d, query_dim=%d, "
        "physical_bytes_per_rank=%d).",
        _workspace.max_rows,
        _workspace.local_heads,
        _workspace.total_heads,
        _workspace.query_dim,
        _workspace.physical_bytes_per_rank,
    )
    return _workspace


def close_dcp_query_vmm_workspace() -> None:
    """Collectively close and reset the singleton workspace."""
    global _workspace, _workspace_failed
    if _workspace is not None:
        _workspace.close()
    _workspace = None
    _workspace_failed = False
