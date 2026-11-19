# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional owner registrations for one-reader gathered-row endpoints."""

import ctypes as c
from pathlib import Path

# Failed deregistration must retain both the registered tables and the library.
_quarantined_owners = []


def load_gather_library(path):
    lib = c.CDLL(str(Path(path).expanduser().resolve(strict=True)))
    u64, u32 = c.c_uint64, c.c_uint
    lib.gg_last_error.restype = c.c_char_p
    lib.gg_owner_create.argtypes = [
        c.c_int,
        c.c_char_p,
        c.c_int,
        u32,
        c.POINTER(u64),
        c.POINTER(u64),
        u32,
        u32,
    ]
    lib.gg_owner_create.restype = c.c_void_p
    lib.gg_owner_close.argtypes = [c.c_void_p]
    lib.gg_owner_close.restype = c.c_int
    return lib


class GatherOwnerRegistrations:
    """Hold one set of table registrations and listener per reader endpoint."""

    def __init__(self, library, tables, gpu, device, gid, ports):
        self.library, self.tables = library, tuple(tables)
        self.handles = []
        ports = tuple(ports)
        if (
            len(self.tables) != 2
            or not ports
            or len(set(ports)) != len(ports)
            or any(not 0 < port <= 65535 for port in ports)
            or any(t.numel() <= 0 or t.numel() % 264 for t in self.tables)
        ):
            raise ValueError("invalid gather tables or endpoint ports")
        addresses = (c.c_uint64 * len(self.tables))(
            *(t.data_ptr() for t in self.tables)
        )
        lengths = (c.c_uint64 * len(self.tables))(*(t.numel() for t in self.tables))
        try:
            for port in ports:
                handle = library.gg_owner_create(
                    gpu,
                    device.encode(),
                    gid,
                    16,
                    addresses,
                    lengths,
                    len(self.tables),
                    port,
                )
                if not handle:
                    raise RuntimeError(self._error("gather owner creation failed"))
                self.handles.append(handle)
        except BaseException:
            try:
                self.close()
            except BaseException:
                _quarantined_owners.append(self)
            raise

    def _error(self, fallback):
        error = self.library.gg_last_error()
        return error.decode(errors="replace") if error else fallback

    def close(self):
        """Close only after readers retire; retain failed registrations."""
        while self.handles:
            if self.library.gg_owner_close(self.handles[-1]):
                if self not in _quarantined_owners:
                    _quarantined_owners.append(self)
                raise RuntimeError(self._error("gather owner retirement failed"))
            self.handles.pop()


def create_gather_checkpoint_owner(args):
    """Create the opt-in owner while reusing the native checkpoint backing."""
    from vllm.distributed.engram_share.rdma.gather_control import GatherOwnerServer
    from vllm.distributed.engram_share.rdma.owner import RealCheckpointOwner

    class GatherCheckpointOwner(RealCheckpointOwner):
        def __init__(self, owner_args):
            super().__init__(owner_args)
            self.gather_registrations = None

        def _start_control(self, metadata):
            ports = range(
                args.gather_base_port, args.gather_base_port + args.gather_readers
            )
            self.gather_registrations = GatherOwnerRegistrations(
                load_gather_library(args.gather_library),
                self.table_views,
                args.owner_gpu,
                args.gather_device,
                args.gather_gid_index,
                ports,
            )
            try:
                return GatherOwnerServer(
                    metadata,
                    [(args.advertise_host, port) for port in ports],
                    host=args.control_host,
                    port=args.control_port,
                ).start()
            except BaseException:
                self.gather_registrations.close()
                raise

        def _retire_additional_registrations(self):
            self.gather_registrations.close()

    return GatherCheckpointOwner(args)
