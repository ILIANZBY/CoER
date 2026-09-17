"""Launch one multi-node torchrun job using Ray Core resource scheduling.

The Ray Job entrypoint itself is CPU-only. It gang-schedules one actor per GPU
node, gives each actor all GPUs on that node, and runs a conventional
multi-node torchrun command inside each actor. This avoids requiring the
optional ``ray[train]`` dependencies on the cluster.
"""

import argparse
import os
import socket
import subprocess
import sys
from collections import Counter

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(max_restarts=0)
class TorchrunNode:
    """Own all GPUs on one node and launch its local torchrun processes."""

    def location(self):
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "node_ip": ray.util.get_node_ip_address(),
            "gpu_ids": list(ray.get_gpu_ids()),
        }

    def choose_master_endpoint(self):
        """Choose an address reachable by workers on the other GPU node."""
        host = ray.util.get_node_ip_address()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return host, port

    def run(
        self,
        *,
        node_rank: int,
        num_nodes: int,
        gpus_per_node: int,
        cpus_per_node: int,
        master_addr: str,
        master_port: int,
        train_script: str,
        training_args: list[str],
    ):
        assigned_gpus = list(ray.get_gpu_ids())
        if len(assigned_gpus) != gpus_per_node:
            raise RuntimeError(
                f"Node rank {node_rank} received {len(assigned_gpus)} GPUs, "
                f"expected {gpus_per_node}: {assigned_gpus}"
            )

        env = os.environ.copy()
        env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("OMP_NUM_THREADS", str(max(1, cpus_per_node // gpus_per_node)))

        preflight = (
            "import torch, transformers, accelerate, deepspeed, flash_attn; "
            "print('torch', torch.__version__, 'cuda', torch.version.cuda); "
            "print('transformers', transformers.__version__); "
            "print('accelerate', accelerate.__version__); "
            "print('deepspeed', deepspeed.__version__); "
            "print('flash_attn', flash_attn.__version__); "
            f"assert torch.cuda.device_count() == {gpus_per_node}, "
            "f'expected assigned GPUs, got {torch.cuda.device_count()}'; "
            "assert torch.cuda.is_bf16_supported(), 'assigned GPUs do not support BF16'"
        )
        subprocess.run([sys.executable, "-c", preflight], env=env, check=True)

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={num_nodes}",
            f"--nproc-per-node={gpus_per_node}",
            f"--node-rank={node_rank}",
            f"--master-addr={master_addr}",
            f"--master-port={master_port}",
            train_script,
            *training_args,
        ]
        print(
            f"Launching torchrun node_rank={node_rank}/{num_nodes - 1} "
            f"on {ray.util.get_node_ip_address()} with {gpus_per_node} GPUs",
            flush=True,
        )
        subprocess.run(command, env=env, check=True)
        return {"node_rank": node_rank, "node_ip": ray.util.get_node_ip_address()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--cpus-per-node", type=int, default=8)
    parser.add_argument("--placement-timeout-seconds", type=int, default=1800)
    parser.add_argument("--train-script", required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.training_args[:1] == ["--"]:
        args.training_args = args.training_args[1:]
    if not args.training_args:
        parser.error("training arguments are required after --")
    if min(args.num_nodes, args.gpus_per_node, args.cpus_per_node) < 1:
        parser.error("node, GPU, and CPU counts must all be positive")
    return args


def main():
    args = parse_args()
    ray.init(address="auto", log_to_driver=True)

    total_gpus = args.num_nodes * args.gpus_per_node
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    if cluster_gpus < total_gpus:
        raise RuntimeError(
            f"Ray cluster advertises {cluster_gpus} GPUs, but this job requires {total_gpus}"
        )

    bundles = [
        {"CPU": args.cpus_per_node, "GPU": args.gpus_per_node}
        for _ in range(args.num_nodes)
    ]
    group = placement_group(bundles, strategy="SPREAD")
    actors = []
    try:
        ray.get(group.ready(), timeout=args.placement_timeout_seconds)
        for bundle_index in range(args.num_nodes):
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_bundle_index=bundle_index,
                placement_group_capture_child_tasks=True,
            )
            actor = TorchrunNode.options(
                num_cpus=args.cpus_per_node,
                num_gpus=args.gpus_per_node,
                scheduling_strategy=strategy,
            ).remote()
            actors.append(actor)

        locations = ray.get([actor.location.remote() for actor in actors])
        node_counts = Counter(location["node_id"] for location in locations)
        print(f"Allocated {total_gpus} GPUs across {len(node_counts)} nodes: {locations}")
        if len(node_counts) != args.num_nodes:
            raise RuntimeError(
                f"Expected {args.num_nodes} distinct GPU nodes, got {len(node_counts)}"
            )

        master_addr, master_port = ray.get(actors[0].choose_master_endpoint.remote())
        runs = [
            actor.run.remote(
                node_rank=node_rank,
                num_nodes=args.num_nodes,
                gpus_per_node=args.gpus_per_node,
                cpus_per_node=args.cpus_per_node,
                master_addr=master_addr,
                master_port=master_port,
                train_script=args.train_script,
                training_args=args.training_args,
            )
            for node_rank, actor in enumerate(actors)
        ]
        results = ray.get(runs)
        print(f"Distributed SFT completed successfully: {results}")
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        remove_placement_group(group)


if __name__ == "__main__":
    main()
