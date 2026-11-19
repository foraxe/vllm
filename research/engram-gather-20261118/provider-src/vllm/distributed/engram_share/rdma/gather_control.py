# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lease distinct single-use gather endpoints from one checkpoint owner."""

from vllm.distributed.engram_share.rdma.owner import CountingOwnerServer
from vllm.distributed.engram_share.rdma.runtime import ControlError


class GatherOwnerServer(CountingOwnerServer):
    """Each prototype gather endpoint accepts exactly one reader, ever.

    A release returns the table lease but cannot make that endpoint reusable:
    the gather listener has already accepted and closed its sole connection.
    """

    def __init__(self, metadata, endpoints, host="127.0.0.1", port=0):
        slots = tuple((h, int(p)) for h, p in endpoints)
        if (
            not slots
            or len(set(slots)) != len(slots)
            or any(
                not isinstance(h, str) or not h or p < 1 or p > 65535 for h, p in slots
            )
        ):
            raise ValueError("gather endpoints must be distinct valid host/port pairs")
        super().__init__(metadata, host, port)
        self._available_gather = list(slots)
        self._gather_leases = {}

    def _dispatch(self, request):
        with self._lock:
            op = request.get("op")
            if op == "attach":
                if not self._available_gather:
                    raise ControlError("no unused gather reader endpoints")
                response = super()._dispatch(request)
                host, port = self._available_gather.pop(0)
                self._gather_leases[response["lease"]] = (host, port)
                response["metadata"]["gather"] = {
                    "protocol": "gather_graph_v1",
                    "host": host,
                    "port": port,
                }
                return response
            if op == "release":
                response = super()._dispatch(request)
                self._gather_leases.pop(request["lease"], None)
                return response
            return super()._dispatch(request)
