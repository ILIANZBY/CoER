import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

# Keep the head-node dependency stack importable; the submitted Ray runtime
# already pins a compatible NumPy/SciPy combination.
if not hasattr(np, "long"):
    np.long = np.int64
if not hasattr(np, "ulong"):
    np.ulong = np.uint64

from verl.workers.rollout import llm_server


class _Replica:
    def __init__(self, *, rank=0, address="old-address", pg="old-pg"):
        self.replica_rank = rank
        self.servers = [f"server-{address}"]
        self.workers = [f"worker-{address}"]
        self.resource_pool = SimpleNamespace(pgs=[pg])
        self._server_handle = self.servers[0]
        self._server_address = address


class _FailingReplacement(_Replica):
    last_instance = None

    def __init__(self, replica_rank, config, model_config, gpus_per_node, name_suffix):
        super().__init__(rank=replica_rank, address="partial-address", pg="partial-pg")
        self.model_config = model_config
        _FailingReplacement.last_instance = self

    async def init_standalone(self):
        raise RuntimeError("injected standalone startup failure")


def test_failed_replica_restart_releases_partial_gpu_resources(monkeypatch):
    killed = []
    removed = []
    load_balancer_removed = []
    monkeypatch.setattr(llm_server.ray, "kill", killed.append)
    monkeypatch.setattr(
        llm_server.ray,
        "get_actor",
        lambda _name: (_ for _ in ()).throw(ValueError("actor not found")),
    )
    monkeypatch.setattr(llm_server.ray.util, "remove_placement_group", removed.append)
    monkeypatch.setattr(
        llm_server.ray.util,
        "placement_group_table",
        lambda _pg: {"state": "REMOVED"},
    )

    manager = object.__new__(llm_server.LLMServerManager)
    old_replica = _Replica()
    manager.rollout_replicas = [old_replica]
    manager.server_handles = [old_replica._server_handle]
    manager.server_addresses = [old_replica._server_address]
    manager.rollout_replica_class = _FailingReplacement
    manager.rollout_config = SimpleNamespace(n_gpus_per_node=2)
    manager.model_config = OmegaConf.create({"path": "base-model"})
    manager.name_suffix = "old_def"

    class _RemoveServers:
        async def remote(self, addresses):
            load_balancer_removed.extend(addresses)

    manager.global_load_balancer = SimpleNamespace(remove_servers=_RemoveServers())

    with pytest.raises(RuntimeError, match="injected standalone startup failure"):
        asyncio.run(
            llm_server.LLMServerManager.restart_replica_with_model.__wrapped__(
                manager, 0, "replacement-model"
            )
        )

    replacement = _FailingReplacement.last_instance
    assert removed == ["old-pg", "partial-pg"]
    assert load_balancer_removed == ["partial-address"]
    assert set(killed) == {
        "server-old-address",
        "worker-old-address",
        "server-partial-address",
        "worker-partial-address",
    }
    assert replacement.resource_pool is None
    assert replacement.servers == []
    assert replacement.workers == []
    # The failed replacement is never published; the same slot remains safe
    # for a later retry after the historical route has been disabled.
    assert manager.rollout_replicas == [old_replica]
    assert manager.server_addresses == ["old-address"]


def test_unconfirmed_pg_removal_keeps_handle_for_retry(monkeypatch):
    monkeypatch.setattr(llm_server.ray, "kill", lambda _actor: None)
    monkeypatch.setattr(llm_server.ray.util, "remove_placement_group", lambda _pg: None)
    monkeypatch.setattr(
        llm_server.ray.util,
        "placement_group_table",
        lambda _pg: {"state": "CREATED"},
    )

    manager = object.__new__(llm_server.LLMServerManager)
    replica = _Replica(pg="still-reserved-pg")
    original_pool = replica.resource_pool
    cleanup_ok = asyncio.run(
        manager._cleanup_standalone_replica_resources(
            replica, placement_group_timeout_s=0.0
        )
    )

    assert cleanup_ok is False
    assert replica.resource_pool is original_pool
    assert replica.resource_pool.pgs == ["still-reserved-pg"]
