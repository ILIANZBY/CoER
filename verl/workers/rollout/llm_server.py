# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


@ray.remote
class GlobalRequestLoadBalancer:
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers."""

    def __init__(self, servers: dict[str, ray.actor.ActorHandle], max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE):
        if not servers:
            raise ValueError("server must be non-empty")

        self._server = servers
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)

    def acquire_server(self, request_id: str) -> str:
        """Acquire a server for the given request, reusing the same server for multi-turn conversations."""
        # request-level sticky (multi-turn: same conversation -> same server)
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            self._inflight_requests[server_id] += 1
            return server_id

        # new request: route to least loaded server
        server_id = min(self._inflight_requests, key=self._inflight_requests.get)
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id

    def release_server(self, server_id: str) -> None:
        """Release a server after a request completes, decrementing its inflight count."""
        if server_id not in self._inflight_requests:
            raise ValueError(f"Invalid server_id for release: {server_id}")
        if self._inflight_requests[server_id] <= 0:
            raise ValueError(f"Release called with no inflight requests on server {server_id}")
        self._inflight_requests[server_id] -= 1

    def add_servers(self, servers: dict[str, ray.actor.ActorHandle]) -> None:
        """Add new servers to the load balancer pool."""
        for sid, handle in servers.items():
            self._server[sid] = handle
            if sid not in self._inflight_requests:
                self._inflight_requests[sid] = 0

    def remove_servers(self, server_ids: list[str]) -> None:
        """Remove servers from the load balancer pool and clean up sticky sessions."""
        for sid in server_ids:
            self._server.pop(sid, None)
            self._inflight_requests.pop(sid, None)
        stale_keys = [k for k, v in self._request_id_to_server.items() if v in server_ids]
        for k in stale_keys:
            del self._request_id_to_server[k]


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        servers: dict[str, ray.actor.ActorHandle],
        load_balancer_handle: ray.actor.ActorHandle,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            servers (dict[str, ray.actor.ActorHandle]): handle for each LLM server.
            load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle: dict[str, ray.actor.ActorHandle] = servers

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        handle = self._server_id_to_handle.get(server_id)
        if handle is None:
            raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        self._load_balancer.release_server.remote(server_id=server_id)

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        server_id, server = await self._acquire_server(request_id)
        try:
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)


class FullyLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        name_suffix: str = "",
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.name_suffix = name_suffix

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.rollout_config.name)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self):
        """Initialize the LLM server replicas."""
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
                name_suffix=self.name_suffix,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            # A full rollout pool can contain many one-GPU vLLM replicas.
            # Starting them strictly one by one leaves most of the reserved
            # GPUs idle for the whole cold-start interval.  Bound concurrency
            # so independent replicas overlap their model load/engine warmup
            # without creating an unbounded storage or CPU burst.
            max_concurrency = max(1, int(self.rollout_config.standalone_init_concurrency))
            max_concurrency = min(max_concurrency, len(self.rollout_replicas))
            try:
                if max_concurrency == 1:
                    for server in self.rollout_replicas:
                        await server.init_standalone()
                else:
                    logger.info(
                        "Initializing %d standalone rollout replicas with concurrency=%d",
                        len(self.rollout_replicas),
                        max_concurrency,
                    )
                    semaphore = asyncio.Semaphore(max_concurrency)

                    async def _init_standalone(server):
                        async with semaphore:
                            await server.init_standalone()

                    await asyncio.gather(*[_init_standalone(server) for server in self.rollout_replicas])
            except BaseException:
                # Standalone startup is not atomic: resource pools, checkpoint
                # workers, and HTTP servers are allocated in that order.  If a
                # later stage fails, explicitly tear down every partially
                # initialized replica so a retry cannot strand reserved GPUs.
                await asyncio.gather(
                    *[
                        self._cleanup_standalone_replica_resources(server)
                        for server in self.rollout_replicas
                    ],
                    return_exceptions=True,
                )
                raise

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        self.global_load_balancer = GlobalRequestLoadBalancer.remote(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )

    def get_client(self, fully_async: bool = False) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            fully_async (bool): Whether to return the FullyLLMServerClient.
        """
        servers = dict(zip(self.server_addresses, self.server_handles, strict=True))
        if not fully_async:
            return LLMServerClient(config=self.config, servers=servers, load_balancer_handle=self.global_load_balancer)
        else:
            return FullyLLMServerClient(
                config=self.config, servers=servers, load_balancer_handle=self.global_load_balancer
            )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])

    async def _cleanup_standalone_replica_resources(
        self,
        replica: RolloutReplica,
        *,
        placement_group_timeout_s: float = 60.0,
    ) -> bool:
        """Best-effort teardown for a complete or partially started replica.

        ``RolloutReplica.init_standalone`` can fail after its placement group or
        only some actors have been created.  Keeping teardown in the manager
        makes both cold-start and population hot-swap failures transactional
        with respect to Ray GPU reservations.
        """
        import time

        for actor in [*getattr(replica, "servers", []), *getattr(replica, "workers", [])]:
            try:
                ray.kill(actor)
            except Exception:
                # The actor may already have died; placement-group removal
                # below is the authoritative resource cleanup.
                logger.warning(
                    "[replica_cleanup] Could not kill an already-failing actor for replica=%s",
                    getattr(replica, "replica_rank", "unknown"),
                    exc_info=True,
                )

        resource_pool = getattr(replica, "resource_pool", None)
        placement_groups = list(getattr(resource_pool, "pgs", None) or [])
        removal_ok = True
        for pg in placement_groups:
            try:
                ray.util.remove_placement_group(pg)
            except Exception:
                removal_ok = False
                logger.warning(
                    "[replica_cleanup] Could not request placement-group removal "
                    "for replica=%s",
                    getattr(replica, "replica_rank", "unknown"),
                    exc_info=True,
                )

        for pg in placement_groups:
            deadline = time.monotonic() + placement_group_timeout_s
            while time.monotonic() < deadline:
                try:
                    state = ray.util.placement_group_table(pg).get("state", "REMOVED")
                except Exception:
                    # GCS no longer knows this placement group, which is
                    # equivalent to removal for this cleanup path.
                    break
                if state == "REMOVED":
                    break
                await asyncio.sleep(0.5)
            else:
                removal_ok = False
                logger.error(
                    "[replica_cleanup] Placement group was not removed within %.1fs "
                    "for replica=%s",
                    placement_group_timeout_s,
                    getattr(replica, "replica_rank", "unknown"),
                )

        # Clear stale actor handles even on a partial teardown.  A subsequent
        # retry must operate on the actors it creates, not on dead actors from
        # the previous attempt.
        replica.servers = []
        replica.workers = []
        # Preserve a handle to a placement group whose removal could not be
        # confirmed.  The manager can then retry cleanup instead of losing the
        # only reference to a still-reserved GPU bundle.
        replica.resource_pool = None if removal_ok else resource_pool
        replica._server_handle = None
        replica._server_address = None
        return removal_ok

    @auto_await
    async def restart_replica_with_model(self, replica_idx: int, new_model_path: str):
        """Destroy a single replica and recreate it with a different model path.

        Used by PopulationManager to swap historical checkpoints on old model replicas.
        The replica's GPU placement group is destroyed and recreated.
        """
        import copy
        import time

        old_replica = self.rollout_replicas[replica_idx]
        old_address = self.server_addresses[replica_idx]

        # Release every old Ray resource before allocating the replacement.
        # A timed-out placement-group removal is fail-closed: allocating on top
        # of it is exactly what causes the repeated "available GPUs 0" loop.
        old_resources_released = await self._cleanup_standalone_replica_resources(old_replica)
        if not old_resources_released:
            raise RuntimeError(
                f"could not fully release old rollout replica {replica_idx}; "
                "refusing to allocate a replacement"
            )

        # Wait for killed actor names to be deregistered in Ray GCS
        name_suffix_str = self.name_suffix.lstrip("_") if self.name_suffix else ""
        expected_suffix = f"_{name_suffix_str}" if name_suffix_str else ""
        server_name = f"vllm_server_{old_replica.replica_rank}_0{expected_suffix}"
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                ray.get_actor(server_name)
                await asyncio.sleep(1)
            except ValueError:
                break
        else:
            logger.warning(f"[restart_replica] Actor {server_name} still exists after 30s, proceeding anyway")

        # Create new replica with updated model path
        updated_model_config = copy.deepcopy(self.model_config)
        from omegaconf import open_dict
        with open_dict(updated_model_config):
            updated_model_config.path = new_model_path

        new_replica = self.rollout_replica_class(
            replica_rank=old_replica.replica_rank,
            config=self.rollout_config,
            model_config=updated_model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
            name_suffix=name_suffix_str,
        )
        try:
            await new_replica.init_standalone()

            # Do not publish the replacement until both server startup and the
            # load-balancer mutation succeed.  On any failure, cleanup below
            # releases the new placement group and the manager retains a dead
            # (but safely retryable) old slot.
            await self.global_load_balancer.remove_servers.remote([old_address])
            await self.global_load_balancer.add_servers.remote(
                {new_replica._server_address: new_replica._server_handle}
            )
        except BaseException:
            if new_replica._server_address is not None:
                try:
                    await self.global_load_balancer.remove_servers.remote(
                        [new_replica._server_address]
                    )
                except Exception:
                    logger.warning(
                        "[restart_replica] Could not remove unpublished replacement "
                        "replica=%d from the load balancer",
                        replica_idx,
                        exc_info=True,
                    )
            cleanup_task = asyncio.create_task(
                self._cleanup_standalone_replica_resources(new_replica)
            )
            cleanup_ok = False
            try:
                cleanup_ok = await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # Preserve cancellation while still waiting for resource
                # release; abandoning cleanup here can reserve GPUs forever.
                cleanup_ok = await cleanup_task
            except Exception:
                logger.exception(
                    "[restart_replica] Failed to clean replacement replica=%d "
                    "after startup failure",
                    replica_idx,
                )
            if not cleanup_ok:
                # Retain the partially initialized object solely so the next
                # retry still has its placement-group handle.  It is not added
                # to routing and its dead endpoint is never published.
                self.rollout_replicas[replica_idx] = new_replica
            raise

        # Publish the fully initialized replacement atomically from the
        # manager's point of view.
        self.rollout_replicas[replica_idx] = new_replica
        self.server_handles[replica_idx] = new_replica._server_handle
        self.server_addresses[replica_idx] = new_replica._server_address
