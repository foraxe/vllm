# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm.distributed.engram_share.rdma.gather_control import GatherOwnerServer
from vllm.distributed.engram_share.rdma.gather_owner import (
    GatherOwnerRegistrations,
    _quarantined_owners,
    create_gather_checkpoint_owner,
)
from vllm.distributed.engram_share.rdma.runtime import ControlError, OwnerClient


def test_distinct_tp4_endpoints_are_never_reissued():
    endpoints = [("127.0.0.1", 41000 + i) for i in range(4)]
    with GatherOwnerServer({"checkpoint_id": "model-a"}, endpoints) as owner:
        host, port = owner.address
        client = OwnerClient(host, port, "model-a", owner.generation)
        with ThreadPoolExecutor(max_workers=4) as pool:
            leases = list(pool.map(lambda _: client.attach(), range(4)))
        assigned = {
            (x.metadata["gather"]["host"], x.metadata["gather"]["port"]) for x in leases
        }
        assert assigned == set(endpoints)
        assert owner.lease_count == 4
        with pytest.raises(ControlError, match="no unused gather"):
            client.attach()
        for lease in leases:
            lease.release()
        assert owner.lease_count == 0
        with pytest.raises(ControlError, match="no unused gather"):
            client.attach()


def test_failed_identity_check_does_not_consume_slot():
    with GatherOwnerServer(
        {"checkpoint_id": "model-a"}, [("localhost", 41000)]
    ) as owner:
        host, port = owner.address
        wrong = OwnerClient(host, port, "model-b", owner.generation)
        with pytest.raises(ControlError, match="checkpoint"):
            wrong.attach()
        correct = OwnerClient(host, port, "model-a", owner.generation)
        lease = correct.attach()
        assert lease.metadata["gather"]["protocol"] == "gather_graph_v1"
        lease.release()


@pytest.mark.parametrize(
    "endpoints", [[], [("host", 1), ("host", 1)], [("", 1)], [("host", 0)]]
)
def test_rejects_invalid_endpoint_sets(endpoints):
    with pytest.raises(ValueError, match="gather endpoints"):
        GatherOwnerServer({"checkpoint_id": "model-a"}, endpoints)


class FakeTable:
    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address

    def numel(self):
        return 2640


class FakeOwnerLibrary:
    def __init__(self, fail_port=None, fail_close=None):
        self.fail_port, self.fail_close = fail_port, fail_close
        self.created, self.closed = [], []

    def gg_owner_create(self, gpu, device, gid, qps, addresses, lengths, n, port):
        assert gpu == 0 and device == b"rdma0" and gid == 3 and qps == 16
        assert list(addresses) == [1000, 2000] and list(lengths) == [2640, 2640]
        assert n == 2
        if port == self.fail_port:
            return None
        self.created.append(port)
        return port

    def gg_owner_close(self, handle):
        if handle == self.fail_close:
            return 1
        self.closed.append(handle)
        return 0

    def gg_last_error(self):
        return b"injected failure"


def test_owner_registers_one_service_per_reader_and_closes_in_reverse():
    api = FakeOwnerLibrary()
    tables = [FakeTable(1000), FakeTable(2000)]
    group = GatherOwnerRegistrations(api, tables, 0, "rdma0", 3, range(41000, 41004))
    assert api.created == [41000, 41001, 41002, 41003]
    assert group.tables == tuple(tables)
    group.close()
    assert api.closed == [41003, 41002, 41001, 41000]


def test_partial_owner_setup_failure_retires_prior_endpoints():
    api = FakeOwnerLibrary(fail_port=41002)
    with pytest.raises(RuntimeError, match="injected failure"):
        GatherOwnerRegistrations(
            api, [FakeTable(1000), FakeTable(2000)], 0, "rdma0", 3, range(41000, 41004)
        )
    assert api.closed == [41001, 41000]


def test_failed_retirement_retains_tables_and_unclosed_handles():
    api = FakeOwnerLibrary(fail_close=41001)
    tables = [FakeTable(1000), FakeTable(2000)]
    group = GatherOwnerRegistrations(api, tables, 0, "rdma0", 3, range(41000, 41002))
    with pytest.raises(RuntimeError, match="injected failure"):
        group.close()
    assert group in _quarantined_owners
    assert group.tables == tuple(tables) and group.handles == [41000, 41001]
    _quarantined_owners.remove(group)


def test_checkpoint_owner_shares_backing_and_drains_before_retirement(monkeypatch):
    api = FakeOwnerLibrary()
    monkeypatch.setattr(
        "vllm.distributed.engram_share.rdma.gather_owner.load_gather_library",
        lambda _: api,
    )
    args = SimpleNamespace(
        gather_base_port=41000,
        gather_readers=4,
        gather_library="unused",
        owner_gpu=0,
        gather_device="rdma0",
        gather_gid_index=3,
        advertise_host="127.0.0.1",
        control_host="127.0.0.1",
        control_port=0,
    )
    owner = create_gather_checkpoint_owner(args)
    owner.table_views = [FakeTable(1000), FakeTable(2000)]
    owner.control = owner._start_control({"checkpoint_id": "model-a"})
    host, port = owner.control.address
    client = OwnerClient(host, port, "model-a", owner.control.generation)
    leases = [client.attach() for _ in range(4)]
    with pytest.raises(ControlError, match="4 lease"):
        owner.control.close()
    assert api.closed == []
    for lease in leases:
        lease.release()
    owner.control.close()
    owner._retire_additional_registrations()
    assert api.closed == [41003, 41002, 41001, 41000]
