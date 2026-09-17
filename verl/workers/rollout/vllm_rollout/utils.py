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
import ctypes
import json
import logging
import os
import platform
import signal
import threading
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )

        received_weight_names: set[str] = set()
        duplicate_weight_names: set[str] = set()
        loaded_param_names: set[str] = set()
        load_result_available = True
        first_load_error: Exception | None = None

        def load_bucket(weights: list[tuple[str, torch.Tensor]]) -> None:
            nonlocal first_load_error, load_result_available
            for name, _ in weights:
                if name in received_weight_names:
                    duplicate_weight_names.add(name)
                received_weight_names.add(name)
            if first_load_error is not None:
                # The REP socket must acknowledge every bucket.  Keep draining
                # the transfer after a loader failure so the sender does not
                # remain blocked forever waiting for an ACK.
                return
            try:
                loaded_params = self._update_weights(
                    weights, peft_config=peft_config, base_sync_done=base_sync_done
                )
            except Exception as exc:
                first_load_error = exc
                return
            if loaded_params is None:
                load_result_available = False
            else:
                loaded_param_names.update(loaded_params)

        receiver.receive_weights(on_bucket_received=load_bucket)

        # Re-raise only after receive_weights has acknowledged the final bucket;
        # otherwise the corresponding sender can deadlock in socket.recv().
        if first_load_error is not None:
            raise RuntimeError(
                "vLLM weight loading failed after safely draining the IPC transfer"
            ) from first_load_error

        # Validate after the final bucket has been acknowledged so a bad load
        # fails the Ray task instead of leaving the sender blocked on ZMQ.
        if use_standard_weight_load:
            self._validate_full_vlm_weight_sync(
                received_weight_names=received_weight_names,
                duplicate_weight_names=duplicate_weight_names,
                loaded_param_names=loaded_param_names,
                load_result_available=load_result_available,
            )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

    def _update_weights(
        self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool
    ) -> set[str] | None:
        weights = self._remap_vlm_weight_names(weights)
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
            return set(weights)
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
                return set(loaded_params)
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                loaded_params = self.model_runner.model.load_weights(weights)
                if loaded_params is not None:
                    logger.info(f"Standard weights loaded (async), loaded_params: {len(loaded_params)}")
                    return set(loaded_params)
                return None

    def _remap_vlm_weight_names(self, weights: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """Remap CausalLM weight names to VLM format if the model uses language_model prefix.

        A text-only Megatron converter can output ``model.layers.*`` (CausalLM
        naming), while a VLM target expects ``language_model.model.layers.*``.
        Full HF VLM exports already use ``model.visual.*``,
        ``model.language_model.*`` and ``lm_head.*``; those names must be left
        for the model's own HF-to-vLLM mapper.  Prefixing them again corrupts
        both the vision and language namespaces.
        """
        model = self.model_runner.model
        if not hasattr(model, "language_model"):
            return weights

        has_hf_vlm_mapper = hasattr(model, "visual") and hasattr(model, "hf_to_vllm_mapper")
        hf_vlm_prefixes = ("model.visual.", "model.language_model.", "lm_head.")
        remapped = []
        for name, tensor in weights:
            if has_hf_vlm_mapper and name.startswith(hf_vlm_prefixes):
                pass
            elif name.startswith("model."):
                name = "language_model." + name
            elif name.startswith("lm_head."):
                name = "language_model." + name
            remapped.append((name, tensor))
        return remapped

    def _validate_full_vlm_weight_sync(
        self,
        received_weight_names: set[str],
        duplicate_weight_names: set[str],
        loaded_param_names: set[str],
        load_result_available: bool,
    ) -> None:
        """Fail closed when a Qwen3.5-VL update does not cover its target weights.

        Some vLLM loaders warn and continue for unknown names.  A successful RPC
        therefore does not by itself mean that the rollout model was updated.
        This validation derives the expected rank-local target names from the
        received HF names, including packed parameters, instead of comparing raw
        source/target counts.  The latter differ legitimately for packed layers
        and frozen visual biases.
        """
        model = self.model_runner.model
        quant_config = getattr(self.model_runner.vllm_config, "quant_config", None)
        is_qwen35_vlm_sync = type(model).__name__ == "Qwen3_5ForConditionalGeneration" and any(
            name.startswith(("model.visual.", "visual.")) for name in received_weight_names
        )
        if not is_qwen35_vlm_sync or quant_config is not None:
            return

        if duplicate_weight_names:
            sample = ", ".join(sorted(duplicate_weight_names)[:20])
            raise RuntimeError(
                "Duplicate tensors in Qwen3.5-VL weight sync: "
                f"{len(duplicate_weight_names)} (first: {sample})"
            )
        if not load_result_available:
            raise RuntimeError("vLLM did not report loaded parameters for a Qwen3.5-VL weight sync")

        from vllm.model_executor.models.utils import is_pp_missing_parameter

        local_param_names = {name for name, _ in model.named_parameters()}
        mapper = model.hf_to_vllm_mapper
        packed_mapping = getattr(model, "packed_modules_mapping", {})
        reverse_packed_mapping = {
            source_name: (target_name, set(source_names))
            for target_name, source_names in packed_mapping.items()
            for source_name in source_names
            if source_name != target_name
        }

        derived_local_names: set[str] = set()
        unexpected_source_names: set[str] = set()
        packed_sources_seen: dict[str, set[str]] = {}
        packed_sources_required: dict[str, set[str]] = {}

        for source_name in received_weight_names:
            mapped_names = mapper.apply_list([source_name])
            if len(mapped_names) != 1:
                unexpected_source_names.add(source_name)
                continue
            mapped_name = mapped_names[0]
            if mapped_name in local_param_names:
                derived_local_names.add(mapped_name)
                continue
            if is_pp_missing_parameter(mapped_name, model):
                continue

            mapped_parts = mapped_name.split(".")
            packed_match = None
            for index, part in enumerate(mapped_parts):
                packed_spec = reverse_packed_mapping.get(part)
                if packed_spec is None:
                    continue
                packed_name, required_sources = packed_spec
                candidate_parts = list(mapped_parts)
                candidate_parts[index] = packed_name
                candidate_name = ".".join(candidate_parts)
                if candidate_name in local_param_names or is_pp_missing_parameter(candidate_name, model):
                    packed_match = (candidate_name, part, required_sources)
                    break

            if packed_match is None:
                unexpected_source_names.add(source_name)
                continue

            candidate_name, source_component, required_sources = packed_match
            if candidate_name in local_param_names:
                derived_local_names.add(candidate_name)
                packed_sources_seen.setdefault(candidate_name, set()).add(source_component)
                packed_sources_required[candidate_name] = required_sources

        source_has_visual_bias = any(
            name.startswith(("model.visual.", "visual.")) and name.endswith(".bias")
            for name in received_weight_names
        )
        required_local_names = {
            name
            for name in local_param_names
            if source_has_visual_bias or not (name.startswith("visual.") and name.endswith(".bias"))
        }
        missing_from_source = required_local_names - derived_local_names
        missing_from_load = derived_local_names - loaded_param_names
        incomplete_packed = {
            name: packed_sources_required[name] - seen
            for name, seen in packed_sources_seen.items()
            if packed_sources_required[name] - seen
        }

        if unexpected_source_names or missing_from_source or missing_from_load or incomplete_packed:
            unexpected_sample = ", ".join(sorted(unexpected_source_names)[:10])
            source_sample = ", ".join(sorted(missing_from_source)[:10])
            load_sample = ", ".join(sorted(missing_from_load)[:10])
            packed_sample = ", ".join(
                f"{name}<-{sorted(parts)}" for name, parts in sorted(incomplete_packed.items())[:10]
            )
            raise RuntimeError(
                "Incomplete Qwen3.5-VL weight sync: "
                f"received={len(received_weight_names)}, derived_local={len(derived_local_names)}, "
                f"loaded={len(loaded_param_names)}, unexpected_source={len(unexpected_source_names)} "
                f"({unexpected_sample}), missing_from_source={len(missing_from_source)} ({source_sample}), "
                f"missing_from_load={len(missing_from_load)} ({load_sample}), "
                f"incomplete_packed={len(incomplete_packed)} ({packed_sample})"
            )

        logger.info(
            "Validated Qwen3.5-VL weight sync: received_source_tensors=%d, loaded_target_params=%d",
            len(received_weight_names),
            len(loaded_param_names),
        )

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.
        Uses replica_rank + local_rank to form handle so it matches the sender side
        regardless of CUDA_VISIBLE_DEVICES differences, and avoids collisions
        when multiple replicas share the same node.
        """
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-replica-{replica_rank}-rank-{self.local_rank}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls
