"""Run InjecAgent base/enhanced OOD evaluation on all Ray GPUs.

Each Ray task owns one GPU, starts a private vLLM replica, and evaluates one
deterministic dataset shard.  Outputs are isolated by model, setting, and shard
so interrupted tasks can safely reuse their own cache without cross-writers.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import ray


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path(
    os.environ.get("CORL_TRAINING_ROOT", EVALUATION_ROOT.parent)
).resolve()
OUT = Path(
    os.environ.get(
        "OOD_OUTPUT_DIR", EVALUATION_ROOT / "results/ood/injecagent"
    )
).resolve()
LOG = Path(
    os.environ.get(
        "OOD_LOG_DIR", EVALUATION_ROOT / "logs/ood/injecagent"
    )
).resolve()
MODELS = {
    "base": Path(os.environ.get("BASE_MODEL_PATH", TRAINING_ROOT / "models/base-model")),
    "corl": Path(os.environ.get("CORL_DEFENDER_PATH", TRAINING_ROOT / "sft_output/defender_sft/checkpoint-360")),
}
SETTINGS = ("base", "enhanced")
NUM_SHARDS = int(os.environ.get("OOD_INJECAGENT_NUM_SHARDS", "3"))
MAX_WORKERS = int(os.environ.get("OOD_INJECAGENT_MAX_WORKERS", "32"))
MAX_ATTEMPTS = int(os.environ.get("OOD_INJECAGENT_MAX_ATTEMPTS", "5"))
ATTEMPT_LABEL = os.environ.get("OOD_ATTEMPT", time.strftime("%Y%m%d_%H%M%S"))


def _validate_settings() -> None:
    if not 1 <= NUM_SHARDS <= 16:
        raise ValueError("OOD_INJECAGENT_NUM_SHARDS must be between 1 and 16")
    if not 1 <= MAX_WORKERS <= 32:
        raise ValueError("OOD_INJECAGENT_MAX_WORKERS must be between 1 and 32")
    if not 1 <= MAX_ATTEMPTS <= 20:
        raise ValueError("OOD_INJECAGENT_MAX_ATTEMPTS must be between 1 and 20")
    for path in MODELS.values():
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path / "config.json")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _expected_count(total: int, shard_index: int, num_shards: int) -> int:
    return len(range(shard_index, total, num_shards))


@ray.remote(num_gpus=1, num_cpus=8, max_retries=0)
def run_shard(label: str, model_path_text: str, setting: str, shard_index: int) -> dict:
    port = _free_port()
    shard_name = f"shard_{shard_index:02d}_of_{NUM_SHARDS:02d}"
    key = f"{label}_{setting}_{shard_name}"
    log_dir = LOG / ATTEMPT_LABEL / key
    result_root = OUT / label / setting / shard_name
    benchmark_result_root = result_root / "injecagent"
    log_dir.mkdir(parents=True, exist_ok=True)
    benchmark_result_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V1": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VERL_QWEN35_TEXT_VLLM_REGISTRY": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "LOCAL_LLM_PORT": str(port),
            "LOCAL_LLM_BASE_URL": f"http://localhost:{port}/v1",
            "LOCAL_LLM_API_KEY": "EMPTY",
            "INJECAGENT_RESULTS_DIR": str(benchmark_result_root),
            "OPENAI_API_KEY": "EMPTY",
            "PYTHONPATH": (
                f"{TRAINING_ROOT / 'AgentDyn/src'}:{TRAINING_ROOT}:"
                f"{EVALUATION_ROOT}:{environment.get('PYTHONPATH', '')}"
            ),
            # The cluster-wide flight-recorder prefix is shared by every Ray
            # task.  Give each vLLM process a private prefix to prevent startup
            # crashes caused by colliding dump_0.pipe files.
            "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": f"/tmp/{key}-{port}-dump_",
            "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": f"/tmp/{key}-{port}-trace_",
        }
    )
    serve_command = [
        sys.executable,
        str(TRAINING_ROOT / "vllm_serve_patched.py"),
        "--model",
        model_path_text,
        "--served-model-name",
        label,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.88",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "32",
        "--dtype",
        "bfloat16",
        "--load-format",
        "safetensors",
        "--trust-request-chat-template",
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--language-model-only",
        "--enforce-eager",
    ]
    server_log_path = log_dir / "vllm.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            serve_command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(240):
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {server.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://localhost:{port}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("vLLM did not become healthy within 8 minutes")

            cwd = EVALUATION_ROOT / "third_party/InjecAgent"
            command = [
                sys.executable,
                "src/evaluate_prompted_agent.py",
                "--model_type",
                "OpenAICompatible",
                "--model_name",
                label,
                "--setting",
                setting,
                "--prompt_type",
                "InjecAgent",
                "--use_cache",
                "--max-workers",
                str(MAX_WORKERS),
                "--num-shards",
                str(NUM_SHARDS),
                "--shard-index",
                str(shard_index),
            ]
            prompted_dir = (
                benchmark_result_root
                / f"prompted_OpenAICompatible_{label}_InjecAgent"
            )
            expected = {
                "dh": _expected_count(510, shard_index, NUM_SHARDS),
                "ds": _expected_count(544, shard_index, NUM_SHARDS),
            }
            return_code = None
            counts = {"dh": 0, "ds": 0}
            benchmark_log_path = log_dir / "benchmark.log"
            for attempt in range(1, MAX_ATTEMPTS + 1):
                with benchmark_log_path.open("a", encoding="utf-8") as run_log:
                    run_log.write(
                        f"\nATTEMPT {attempt}/{MAX_ATTEMPTS}: {' '.join(command)}\n"
                    )
                    run_log.flush()
                    return_code = subprocess.run(
                        command,
                        cwd=cwd,
                        env=environment,
                        stdout=run_log,
                        stderr=subprocess.STDOUT,
                    ).returncode
                counts = {
                    attack: _line_count(
                        prompted_dir / f"test_cases_{attack}_{setting}.json"
                    )
                    for attack in ("dh", "ds")
                }
                if return_code == 0 and counts == expected:
                    return {
                        "model": label,
                        "setting": setting,
                        "shard_index": shard_index,
                        "num_shards": NUM_SHARDS,
                        "state": "complete",
                        "attempts": attempt,
                        "counts": counts,
                        "expected": expected,
                        "result_root": str(result_root),
                        "log_dir": str(log_dir),
                    }
                time.sleep(min(5 * attempt, 30))
            raise RuntimeError(
                f"benchmark incomplete after {MAX_ATTEMPTS} attempts: "
                f"return_code={return_code}, counts={counts}, expected={expected}"
            )
        finally:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _write_status(status: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / f".status.{os.getpid()}.json"
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUT / "status.json")


def main() -> None:
    _validate_settings()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    ray.init(address="auto", logging_level="ERROR")
    refs: dict[object, str] = {}
    status: dict[str, dict] = {}
    for shard_index in range(NUM_SHARDS):
        for setting in SETTINGS:
            for label, model_path in MODELS.items():
                key = f"{label}/{setting}/shard_{shard_index:02d}_of_{NUM_SHARDS:02d}"
                reference = run_shard.remote(
                    label, str(model_path), setting, shard_index
                )
                refs[reference] = key
                status[key] = {"state": "submitted"}
    _write_status(status)

    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=30)
        if not ready:
            continue
        reference = ready[0]
        key = refs[reference]
        try:
            status[key] = ray.get(reference)
        except Exception as error:
            status[key] = {
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        _write_status(status)


if __name__ == "__main__":
    main()
