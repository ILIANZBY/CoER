"""Co-Training Orchestrator — top-level coordinator for attacker-defender self-play.

Manages:
- 2 FullyAsyncTrainers (attacker PPO + defender PPO)
- 1 CoTrainRollouter (unified rollout with 2 or 4 model groups)
- 2 MessageQueues (attacker_mq + defender_mq)
- optionally 1 PopulationManager (periodic historical-checkpoint refresh)
- Weight synchronization (NCCL) from trainers to current vLLM instances

Usage:
    python3 cotrain/orchestrator.py --config-path=config --config-name=cotrain_config.yaml
"""

import asyncio
import copy
import json
import logging
import os
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.separation.utils import create_resource_pool_manager, create_role_worker_mapping
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from cotrain.rollouter import CoTrainRollouter
from cotrain.old_model_manager import PopulationManager

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class CoTrainOrchestrator:
    """Coordinate bilateral Co-PPO and dynamic historical populations."""

    def __init__(self):
        self.running = False
        self.components = {}
        self.shutdown_event = threading.Event()

    def run(self, config):
        logger.info("[CO-TRAIN] Starting attacker-defender training...")
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        self._initialize_components(config)
        self._run_training_loop()

    def _initialize_components(self, config):
        logger.info(f"[CO-TRAIN] Orchestrator hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        cotrain_cfg = config.cotrain
        attacker_model_path = cotrain_cfg.attacker_model_path
        defender_model_path = cotrain_cfg.defender_model_path
        population_enabled = bool(cotrain_cfg.get("population_enabled", True))
        training_mode = str(cotrain_cfg.get("training_mode", "dual"))
        if training_mode != "dual":
            raise ValueError(
                f"Co-PPO requires bilateral training ('dual'), got {training_mode!r}"
            )

        self.components["training_mode"] = training_mode
        self.components["population_enabled"] = population_enabled
        logger.info(
            "[CO-TRAIN] training_mode=%s population_enabled=%s",
            training_mode,
            population_enabled,
        )

        # Load role-specific tokenizers/processors.  Sharing the attacker
        # tokenizer with the defender can silently change chat framing, EOS,
        # padding, or added-token semantics even when the base vocab matches.
        logger.info("[CO-TRAIN] Loading role-specific tokenizers and processors...")
        attacker_local_path = copy_to_local(
            attacker_model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        defender_local_path = copy_to_local(
            defender_model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        attacker_tokenizer = hf_tokenizer(attacker_local_path, trust_remote_code=trust_remote_code)
        attacker_processor = hf_processor(attacker_local_path, trust_remote_code=trust_remote_code, use_fast=True)
        defender_tokenizer = hf_tokenizer(defender_local_path, trust_remote_code=trust_remote_code)
        defender_processor = hf_processor(defender_local_path, trust_remote_code=trust_remote_code, use_fast=True)

        self.components["attacker_tokenizer"] = attacker_tokenizer
        self.components["attacker_processor"] = attacker_processor
        self.components["defender_tokenizer"] = defender_tokenizer
        self.components["defender_processor"] = defender_processor
        self.components["config"] = config

        attacker_config = self._build_attacker_config(config)
        defender_config = self._build_defender_config(config)

        # Create role worker mapping
        logger.info("[CO-TRAIN] Creating worker mappings...")
        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # Reserve configured training pools before allocating standalone
        # rollout replicas. Otherwise rollout placement groups can fragment
        # every node and starve STRICT_PACK trainer pools.
        from concurrent.futures import ThreadPoolExecutor

        # Reserve trainer nodes before allocating standalone rollout pools.
        component_specs = [
            (self._create_trainer, ("defender", defender_config)),
            (self._create_rollouter, (config,)),
        ]
        component_specs.insert(
            0, (self._create_trainer, ("attacker", attacker_config))
        )
        logger.info(
            "[CO-TRAIN] Creating components in parallel: %s",
            ["attacker_trainer", "defender_trainer", "rollouter"],
        )
        with ThreadPoolExecutor(max_workers=len(component_specs)) as component_pool:
            component_futures = [
                component_pool.submit(factory, *args)
                for factory, args in component_specs
            ]
            for future in component_futures:
                future.result()

        logger.info("[CO-TRAIN] Component actors created successfully")

        # Training pools are now pinned to complete nodes.  Start the current
        # rollout groups, plus the two historical groups only when population
        # training is enabled.  The no-population ablation therefore does not
        # reserve idle old-model GPUs or create frozen serving replicas.
        rollout_configs = {
            "def": self._build_rollout_config(config, role="defender"),
        }
        rollout_configs["atk"] = self._build_rollout_config(
            config, role="attacker"
        )
        if population_enabled:
            rollout_configs.update(
                {
                    "old_atk": self._build_rollout_config(config, role="old_attacker"),
                    "old_def": self._build_rollout_config(config, role="old_defender"),
                }
            )

        from verl.workers.rollout.llm_server import LLMServerManager

        def _create_mgr(name, cfg):
            logger.info(f"[CO-TRAIN] Creating LLMServerManager: {name}")
            return name, LLMServerManager.create(config=cfg, name_suffix=name)

        def _create_all_managers():
            mgrs = {}
            with ThreadPoolExecutor(max_workers=len(rollout_configs)) as manager_pool:
                futures = [
                    manager_pool.submit(_create_mgr, name, rollout_config)
                    for name, rollout_config in rollout_configs.items()
                ]
                for future in futures:
                    name, manager = future.result()
                    mgrs[name] = manager
            return mgrs

        # Create the elite historical population only for population runs.
        if population_enabled:
            logger.info("[CO-TRAIN] Creating PopulationManager...")
            population_manager = PopulationManager(
                attacker_ckpt_dir=cotrain_cfg.get("attacker_ckpt_dir", config.trainer.default_local_dir),
                defender_ckpt_dir=cotrain_cfg.get("defender_ckpt_dir", config.trainer.default_local_dir),
                population_size=cotrain_cfg.get("population_size", 4),
                update_interval=cotrain_cfg.get("population_update_interval", 20),
                save_freq=config.trainer.save_freq,
                min_eval_samples_per_bucket=cotrain_cfg.get(
                    "population_min_eval_samples_per_bucket", 32
                ),
                min_eval_buckets=cotrain_cfg.get("population_min_eval_buckets", 7),
                required_buckets=cotrain_cfg.get(
                    "population_required_buckets",
                    "slack,shopping,workspace,dailylife,banking,github,travel",
                ),
                probation_sampling_prob=cotrain_cfg.get(
                    "population_probation_sampling_prob", 0.75
                ),
                candidate_lag_steps=cotrain_cfg.get(
                    "population_candidate_lag_steps", 0
                ),
                random_seed=cotrain_cfg.get("population_random_seed", 0),
            )
            self.components["old_model_manager"] = population_manager
        else:
            logger.info(
                "[CO-TRAIN] Population disabled: historical managers and replicas will not be created"
            )

        # Sync total_train_steps
        total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
        logger.info(f"[CO-TRAIN] total_train_steps: {total_train_steps}")
        ray.get(
            self.components["attacker_trainer"].set_total_train_steps.remote(
                total_train_steps
            )
        )
        ray.get(self.components["defender_trainer"].set_total_train_steps.remote(total_train_steps))

        max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
        logger.info(
            "[CO-TRAIN] Creating %s MessageQueue(s) (max_size=%s)...",
            2,
            max_queue_size,
        )

        attacker_mq = MessageQueue.remote(config, max_queue_size)
        attacker_mq_client = MessageQueueClient(attacker_mq)
        defender_mq = MessageQueue.remote(config, max_queue_size)
        defender_mq_client = MessageQueueClient(defender_mq)

        self.components["attacker_mq"] = attacker_mq
        self.components["attacker_mq_client"] = attacker_mq_client
        self.components["defender_mq"] = defender_mq
        self.components["defender_mq_client"] = defender_mq_client

        # Wire MQ clients
        ray.get(self.components["rollouter"].set_mq_clients.remote(attacker_mq_client, defender_mq_client))
        ray.get(
            self.components["attacker_trainer"].set_message_queue_client.remote(
                attacker_mq_client
            )
        )
        ray.get(self.components["defender_trainer"].set_message_queue_client.remote(defender_mq_client))

        # Wire rollouter reference for reset_staleness
        ray.get(
            self.components["attacker_trainer"].set_rollouter.remote(
                self.components["rollouter"]
            )
        )
        ray.get(self.components["defender_trainer"].set_rollouter.remote(self.components["rollouter"]))

        # Set cotrain_role so trainers pass their identity to reset_staleness
        ray.get(
            self.components["attacker_trainer"].set_cotrain_role.remote(
                "attacker"
            )
        )
        ray.get(self.components["defender_trainer"].set_cotrain_role.remote("defender"))

        # Restore trainer process groups deterministically before vLLM starts
        # creating its own CheckpointEngine NCCL groups.  Concurrent dist-ckpt
        # restore and four-pool vLLM startup has caused native Ray/NCCL
        # segfaults during actor checkpoint loading.  This ordering only
        # affects startup; all four rollout pools still initialize in parallel
        # and steady-state training continues to use every configured GPU.
        logger.info("[CO-TRAIN] Loading attacker checkpoint before rollout NCCL startup...")
        attacker_version = ray.get(
            self.components["attacker_trainer"].load_checkpoint.remote()
        )
        logger.info("[CO-TRAIN] Loading defender checkpoint before rollout NCCL startup...")
        defender_version = ray.get(
            self.components["defender_trainer"].load_checkpoint.remote()
        )
        ray.get(self.components["rollouter"].load_checkpoint.remote())
        ray.get(
            self.components["rollouter"].reconcile_policy_versions.remote(
                attacker_param_version=attacker_version,
                defender_param_version=defender_version,
            )
        )
        logger.info("[CO-TRAIN] Checkpoints loaded. Starting configured rollout pools in parallel...")
        mgrs = _create_all_managers()
        atk_llm_mgr = mgrs.get("atk")
        def_llm_mgr = mgrs["def"]

        if atk_llm_mgr is not None:
            self.components["attacker_llm_server_manager"] = atk_llm_mgr
        self.components["defender_llm_server_manager"] = def_llm_mgr

        # Override each trainer's CheckpointEngineManager with its own replicas
        atk_replicas = atk_llm_mgr.get_replicas() if atk_llm_mgr is not None else []
        def_replicas = def_llm_mgr.get_replicas()
        logger.info(f"[CO-TRAIN] Attacker replicas: {len(atk_replicas)}, Defender replicas: {len(def_replicas)}")

        ray.get(
            self.components["attacker_trainer"].setup_checkpoint_manager_with_replicas.remote(
                atk_replicas
            )
        )
        ray.get(self.components["defender_trainer"].setup_checkpoint_manager_with_replicas.remote(def_replicas))

        # Frozen attacker replicas are restored from checkpoint files below;
        # only current-policy serving groups participate in NCCL sync.
        logger.info("[CO-TRAIN] Initial NCCL weight sync to current vLLM replicas...")
        sync_futures = [
            self.components["defender_trainer"]._fit_update_weights.remote()
        ]
        sync_futures.append(
            self.components["attacker_trainer"]._fit_update_weights.remote()
        )
        ray.get(sync_futures)

        # Set model URLs on rollouter
        current_atk_urls = (
            [f"http://{addr}/v1" for addr in atk_llm_mgr.get_addresses()]
            if atk_llm_mgr is not None
            else []
        )
        current_def_urls = [f"http://{addr}/v1" for addr in def_llm_mgr.get_addresses()]
        ray.get(self.components["rollouter"].set_current_model_urls.remote(current_atk_urls, current_def_urls))
        old_atk_urls: list[str] = []
        old_def_urls: list[str] = []
        if population_enabled:
            old_atk_llm_mgr = mgrs["old_atk"]
            old_def_llm_mgr = mgrs.get("old_def")
            self.components["old_attacker_llm_server_manager"] = old_atk_llm_mgr
            if old_def_llm_mgr is not None:
                self.components["old_defender_llm_server_manager"] = old_def_llm_mgr
            old_atk_urls = [f"http://{addr}/v1" for addr in old_atk_llm_mgr.get_addresses()]
            old_def_urls = (
                [f"http://{addr}/v1" for addr in old_def_llm_mgr.get_addresses()]
                if old_def_llm_mgr is not None
                else []
            )
            ray.get(
                self.components["rollouter"].set_old_model_urls.remote(
                    old_atk_urls, old_def_urls
                )
            )

            # Wire PopulationManager to LLMServerManagers and rollouter.
            population_manager = self.components["old_model_manager"]
            population_manager.set_llm_managers(old_atk_llm_mgr, old_def_llm_mgr)
            old_atk_urls, old_def_urls = ray.get(
                self.components["rollouter"].set_old_model_manager.remote(
                    population_manager
                )
            )

        # Warmup: send a dummy request to each vLLM to wake from sleep mode
        logger.info("[CO-TRAIN] Warming up vLLM servers...")
        self._warmup_vllm_servers(current_atk_urls + current_def_urls + old_atk_urls + old_def_urls)

        logger.info("[CO-TRAIN] All components initialized successfully")

    def _warmup_vllm_servers(self, urls: list[str]):
        """Warm every server and fail early unless exact PPO traces are supported."""
        import httpx

        # Exercise the same request-scoped template path used by both agent
        # loops.  A plain warmup request is insufficient: vLLM can answer it
        # successfully while rejecting the actual Defender request because
        # trust_request_chat_template was not enabled.
        request_chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' ~ message['role'] ~ '\\n' ~ message['content']|trim ~ '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

        def _ping(url: str):
            base = url.rstrip("/")
            try:
                # First get actual model name from /models endpoint
                models_resp = httpx.get(f"{base}/models", timeout=30.0)
                model_name = "default"
                if models_resp.status_code == 200:
                    data = models_resp.json().get("data", [])
                    if data:
                        model_name = data[0].get("id", "default")

                resp = httpx.post(
                    f"{base}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Reply OK."}],
                        "max_tokens": 2,
                        "min_tokens": 1,
                        "temperature": 0,
                        "logprobs": True,
                        "return_token_ids": True,
                        "chat_template": request_chat_template,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                    timeout=600.0,
                )
                resp.raise_for_status()
                body = resp.json()
                choices = body.get("choices") or []
                choice = choices[0] if choices else {}
                prompt_token_ids = body.get("prompt_token_ids")
                token_ids = choice.get("token_ids")
                logprob_content = (choice.get("logprobs") or {}).get("content")
                if not isinstance(prompt_token_ids, list):
                    raise RuntimeError("response is missing prompt_token_ids")
                if not isinstance(token_ids, list):
                    raise RuntimeError("response is missing choices[0].token_ids")
                if not isinstance(logprob_content, list):
                    raise RuntimeError("response is missing choices[0].logprobs.content")
                if len(token_ids) != len(logprob_content):
                    raise RuntimeError(
                        "token/logprob length mismatch during preflight: "
                        f"{len(token_ids)} != {len(logprob_content)}"
                    )
                logger.info(
                    "[WARMUP] %s model=%s exact-token preflight OK "
                    "(prompt=%d, generated=%d)",
                    base,
                    model_name,
                    len(prompt_token_ids),
                    len(token_ids),
                )
                return None
            except Exception as e:
                return f"{base}: {e}"

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            errors = [error for error in pool.map(_ping, urls) if error]
        if errors:
            raise RuntimeError(
                "vLLM exact-token preflight failed; adv-evo PPO will not start with "
                "ambiguous token/logprob traces:\n  " + "\n  ".join(errors)
            )

    def _build_attacker_config(self, base_config):
        """Build the adaptive attacker configuration for trajectory-level PPO."""
        cfg = copy.deepcopy(base_config)
        with open_dict(cfg):
            cfg.actor_rollout_ref.model.path = cfg.cotrain.attacker_model_path
            attacker_algorithm = str(cfg.cotrain.get("attacker_algorithm", "ppo")).lower()
            if attacker_algorithm != "ppo":
                raise ValueError(f"Unsupported cotrain.attacker_algorithm={attacker_algorithm!r}")

            cfg.algorithm.use_kl_in_reward = False
            cfg.algorithm.kl_ctrl.kl_coef = 0.0
            # bypass_mode=True: consume the exact log probabilities collected
            # for each adaptive attacker generation.
            cfg.algorithm.rollout_correction = {"bypass_mode": True}

            cfg.algorithm.adv_estimator = "gae"
            cfg.algorithm.gamma = cfg.cotrain.get("attacker_gamma", 1.0)
            cfg.algorithm.lam = cfg.cotrain.get("attacker_lam", 0.95)
            cfg.algorithm.norm_adv_by_std_in_grpo = False
            cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "vanilla"
            if hasattr(cfg.algorithm, "filter_groups"):
                cfg.algorithm.filter_groups = {"enable": False}

            cfg.critic.enable = True
            cfg.critic.strategy = "megatron"
            cfg.critic.model.path = cfg.cotrain.attacker_model_path
            cfg.critic.model.trust_remote_code = True
            cfg.critic.model.enable_gradient_checkpointing = True
            cfg.critic.optim.lr = cfg.cotrain.get("attacker_critic_lr", 1e-5)
            cfg.critic.optim.lr_warmup_steps = cfg.cotrain.get("attacker_critic_lr_warmup", 20)
            cfg.critic.optim.lr_decay_style = "constant"
            cfg.critic.optim.weight_decay = 0.1
            cfg.critic.ppo_mini_batch_size = cfg.actor_rollout_ref.actor.ppo_mini_batch_size
            cfg.critic.ppo_micro_batch_size_per_gpu = 1
            cfg.critic.ppo_max_token_len_per_gpu = cfg.cotrain.get(
                "attacker_critic_token_len_per_gpu", 32768
            )
            cfg.critic.forward_max_token_len_per_gpu = cfg.cotrain.get(
                "attacker_critic_token_len_per_gpu", 32768
            )
            cfg.critic.ppo_infer_max_token_len_per_gpu = cfg.cotrain.get(
                "attacker_token_len_per_gpu", 32768
            )
            cfg.critic.ppo_infer_micro_batch_size_per_gpu = 1
            cfg.critic.use_dynamic_bsz = True
            cfg.critic.megatron.use_mbridge = True
            cfg.critic.megatron.vanilla_mbridge = True
            cfg.critic.megatron.use_remove_padding = True
            cfg.critic.megatron.tensor_model_parallel_size = (
                cfg.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
            )
            cfg.critic.megatron.pipeline_model_parallel_size = 1
            cfg.critic.megatron.context_parallel_size = cfg.cotrain.get("attacker_cp", 1)
            cfg.critic.megatron.expert_model_parallel_size = 1
            cfg.critic.megatron.expert_tensor_parallel_size = 1
            cfg.critic.megatron.param_offload = cfg.actor_rollout_ref.actor.megatron.param_offload
            cfg.critic.megatron.optimizer_offload = (
                cfg.actor_rollout_ref.actor.megatron.optimizer_offload
            )
            cfg.critic.megatron.grad_offload = cfg.actor_rollout_ref.actor.megatron.grad_offload
            cfg.critic.megatron.dtype = "bfloat16"
            cfg.critic.megatron.override_transformer_config = {
                "recompute_method": "uniform",
                "recompute_granularity": "full",
                "recompute_num_layers": 1,
                "attention_backend": "flash",
            }

            attacker_kl_loss_coef = float(cfg.cotrain.get("attacker_kl_loss_coef", 0.001))
            cfg.actor_rollout_ref.actor.use_kl_loss = attacker_kl_loss_coef > 0.0
            cfg.actor_rollout_ref.actor.kl_loss_coef = attacker_kl_loss_coef
            cfg.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
            # VAPO/DAPO-style asymmetric clipping: keep the conservative
            # lower bound while allowing successful attacker updates a
            # little more room on the positive side.
            cfg.actor_rollout_ref.actor.clip_ratio = 0.2
            cfg.actor_rollout_ref.actor.clip_ratio_low = float(
                cfg.cotrain.get("attacker_clip_ratio_low", 0.2)
            )
            cfg.actor_rollout_ref.actor.clip_ratio_high = float(
                cfg.cotrain.get("attacker_clip_ratio_high", 0.28)
            )
            cfg.trainer.critic_warmup = cfg.cotrain.get("attacker_critic_warmup", 40)

            # Unique NCCL group name to avoid collision with defender
            cfg.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs = {"nccl": {"group_name": "atk_ckpt"}}

            # Role-specific sequence and optimization settings
            cfg.data.max_prompt_length = cfg.cotrain.get("attacker_max_prompt_length", 8192)
            cfg.data.max_response_length = cfg.cotrain.get("attacker_max_response_length", 49152)
            cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = cfg.cotrain.get(
                "attacker_token_len_per_gpu", 32768
            )
            cfg.actor_rollout_ref.actor.optim.lr = cfg.cotrain.get("attacker_lr", 5e-7)
            cfg.actor_rollout_ref.actor.optim.lr_decay_style = "constant"
            cfg.actor_rollout_ref.actor.optim.lr_decay_steps = 400
            # Trainer naming — same project as base, different experiment/run name
            cfg.trainer.project_name = base_config.trainer.project_name
            cfg.trainer.experiment_name = cfg.cotrain.get("attacker_exp_name", "cotrain_attacker_ppo")
            cfg.trainer.sibling_run = cfg.cotrain.get("defender_exp_name", "cotrain_defender_ppo")
            cfg.trainer.nnodes = int(
                cfg.cotrain.get("attacker_train_nnodes", cfg.trainer.nnodes)
            )
            if cfg.trainer.nnodes <= 0:
                raise ValueError("cotrain.attacker_train_nnodes must be positive")

            # Checkpoint dir
            if cfg.cotrain.get("attacker_ckpt_dir"):
                cfg.trainer.default_local_dir = cfg.cotrain.attacker_ckpt_dir

            # The text-only Qwen3.5 import has a CP-local ``dt_bias`` mismatch.
            # CP=1 keeps the GatedDeltaNet head layout consistent; DP=4 still
            # uses all eight attacker training GPUs.
            cfg.actor_rollout_ref.actor.megatron.context_parallel_size = cfg.cotrain.get(
                "attacker_cp", 1
            )

        return cfg

    def _build_defender_config(self, base_config):
        """Build defender-specific config: PPO with GAE + Critic + KL loss.

        Full PPO training with value function (critic) for stable advantage
        estimation via GAE. KL loss prevents the defender from drifting too far
        from the reference policy (important for maintaining utility on clean tasks).
        """
        cfg = copy.deepcopy(base_config)
        with open_dict(cfg):
            cfg.actor_rollout_ref.model.path = cfg.cotrain.defender_model_path

            # PPO with GAE (requires critic)
            cfg.algorithm.adv_estimator = "gae"
            cfg.algorithm.gamma = 1.0
            cfg.algorithm.lam = 0.95
            cfg.actor_rollout_ref.actor.policy_loss.loss_mode = "vanilla"
            cfg.algorithm.use_kl_in_reward = False
            cfg.algorithm.kl_ctrl.kl_coef = 0.0
            # bypass_mode=True: use rollout_log_probs collected during generation
            cfg.algorithm.rollout_correction = {"bypass_mode": True}
            # Defender uses PPO (not DAPO) — disable GRPO-specific filter_groups
            if hasattr(cfg.algorithm, "filter_groups"):
                cfg.algorithm.filter_groups = {"enable": False}

            # Enable critic (value function for GAE)
            cfg.critic.enable = True
            cfg.critic.strategy = "megatron"
            cfg.critic.model.path = cfg.cotrain.defender_model_path
            cfg.critic.model.trust_remote_code = True
            cfg.critic.model.enable_gradient_checkpointing = True
            cfg.critic.optim.lr = cfg.cotrain.get("defender_critic_lr", 1e-5)
            cfg.critic.optim.lr_warmup_steps = cfg.cotrain.get(
                "defender_critic_lr_warmup", 40
            )
            cfg.critic.optim.lr_decay_style = "constant"
            cfg.critic.optim.weight_decay = 0.1
            cfg.critic.ppo_mini_batch_size = cfg.actor_rollout_ref.actor.ppo_mini_batch_size
            cfg.critic.ppo_micro_batch_size_per_gpu = 2
            cfg.critic.ppo_max_token_len_per_gpu = cfg.cotrain.get(
                "defender_critic_token_len_per_gpu", 32768
            )
            cfg.critic.forward_max_token_len_per_gpu = cfg.cotrain.get(
                "defender_critic_token_len_per_gpu", 32768
            )
            cfg.critic.ppo_infer_max_token_len_per_gpu = cfg.cotrain.get(
                "defender_token_len_per_gpu", 32768
            )
            cfg.critic.ppo_infer_micro_batch_size_per_gpu = 2
            cfg.critic.use_dynamic_bsz = True
            cfg.critic.megatron.use_mbridge = True
            cfg.critic.megatron.vanilla_mbridge = True
            cfg.critic.megatron.use_remove_padding = True
            cfg.critic.megatron.tensor_model_parallel_size = 2
            cfg.critic.megatron.pipeline_model_parallel_size = 1
            cfg.critic.megatron.context_parallel_size = cfg.cotrain.get("defender_cp", 4)
            cfg.critic.megatron.expert_model_parallel_size = 1
            cfg.critic.megatron.expert_tensor_parallel_size = 1
            cfg.critic.megatron.param_offload = True
            cfg.critic.megatron.optimizer_offload = True
            cfg.critic.megatron.grad_offload = True
            cfg.critic.megatron.dtype = "bfloat16"
            cfg.critic.megatron.override_transformer_config = {
                "recompute_method": "uniform",
                "recompute_granularity": "full",
                "recompute_num_layers": 1,
                "attention_backend": "flash",
            }

            # Unique NCCL group name to avoid collision with attacker
            cfg.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs = {"nccl": {"group_name": "def_ckpt"}}

            # Defender-specific: longer sequences, lower LR
            cfg.data.max_prompt_length = cfg.cotrain.get("defender_max_prompt_length", 20240)
            cfg.data.max_response_length = cfg.cotrain.get("defender_max_response_length", 20240)
            cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = cfg.cotrain.get(
                "defender_token_len_per_gpu", 32768
            )
            cfg.actor_rollout_ref.actor.optim.lr = cfg.cotrain.get("defender_lr", 5e-7)
            cfg.actor_rollout_ref.actor.optim.lr_decay_style = "constant"
            cfg.actor_rollout_ref.actor.optim.lr_decay_steps = 400

            # KL loss to keep defender conservative
            cfg.actor_rollout_ref.actor.use_kl_loss = True
            cfg.actor_rollout_ref.actor.kl_loss_coef = 0.001
            cfg.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"

            # Role-specific asymmetric clipping.  The defender starts more
            # conservatively than the attacker because its resistance reward
            # is much denser and aggressive updates can overfit to the current
            # attack population.
            cfg.actor_rollout_ref.actor.clip_ratio = 0.2
            cfg.actor_rollout_ref.actor.clip_ratio_low = float(
                cfg.cotrain.get("defender_clip_ratio_low", 0.2)
            )
            cfg.actor_rollout_ref.actor.clip_ratio_high = float(
                cfg.cotrain.get("defender_clip_ratio_high", 0.2)
            )

            # Critic warmup: first N steps only train critic.  Recovery jobs
            # can additionally request checkpoint-relative recalibration.
            cfg.trainer.critic_warmup = cfg.cotrain.get("defender_critic_warmup", 60)

            # Trainer naming — same project as base, different experiment/run name
            cfg.trainer.project_name = base_config.trainer.project_name
            cfg.trainer.experiment_name = cfg.cotrain.get("defender_exp_name", "cotrain_defender_ppo")
            cfg.trainer.sibling_run = cfg.cotrain.get("attacker_exp_name", "cotrain_attacker_ppo")
            cfg.trainer.nnodes = int(
                cfg.cotrain.get("defender_train_nnodes", cfg.trainer.nnodes)
            )
            if cfg.trainer.nnodes <= 0:
                raise ValueError("cotrain.defender_train_nnodes must be positive")

            # Checkpoint dir
            if cfg.cotrain.get("defender_ckpt_dir"):
                cfg.trainer.default_local_dir = cfg.cotrain.defender_ckpt_dir

            # Megatron for defender: TP=2, CP=4 (long context 40k)
            cfg.actor_rollout_ref.actor.megatron.context_parallel_size = cfg.cotrain.get("defender_cp", 4)

        return cfg

    def _create_trainer(self, name: str, config, max_retries: int = 3):
        """Create and initialize a FullyAsyncTrainer with retry on port conflicts."""
        import time as _time

        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        # Remove Critic role if not needed (attacker uses GRPO, no critic)
        if not need_critic(config) and Role.Critic in trainer_role_mapping:
            del trainer_role_mapping[Role.Critic]

        # Add RefPolicy if needed but missing (defender uses KL loss)
        if need_reference_policy(config) and Role.RefPolicy not in trainer_role_mapping:
            from verl.experimental.separation.engine_workers import DetachActorWorker
            trainer_role_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

        last_error = None
        for attempt in range(max_retries):
            # Use trainer-name-prefixed pool to avoid PG name collisions between attacker/defender
            resource_pool_manager = self._create_named_resource_pool_manager(name, config, list(trainer_role_mapping.keys()))

            trainer = FullyAsyncTrainer.remote(
                config=config,
                tokenizer=self.components[f"{name}_tokenizer"],
                role_worker_mapping=trainer_role_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=self.components["ray_worker_group_cls"],
                processor=self.components[f"{name}_processor"],
                device_name=config.trainer.device,
            )
            try:
                ray.get(trainer.init_workers.remote())
                self.components[f"{name}_trainer"] = trainer
                logger.info(f"[CO-TRAIN] {name} trainer created successfully")
                return
            except Exception as e:
                last_error = e
                error_str = str(e)
                if "EADDRINUSE" in error_str or "address already in use" in error_str:
                    logger.warning(
                        f"[CO-TRAIN] {name} trainer init_workers failed (attempt {attempt + 1}/{max_retries}) "
                        f"due to port conflict, retrying in 5s..."
                    )
                    try:
                        ray.kill(trainer, no_restart=True)
                    except Exception as kill_error:
                        logger.warning(f"[CO-TRAIN] Failed to kill failed {name} trainer: {kill_error}")
                    if attempt < max_retries - 1:
                        _time.sleep(5)
                else:
                    raise

        raise RuntimeError(
            f"[CO-TRAIN] {name} trainer failed after {max_retries} retries: {last_error}"
        )

    def _create_named_resource_pool_manager(self, name: str, config, roles: list):
        """Create a ResourcePoolManager with a unique pool name prefix per trainer."""
        import time

        from verl.single_controller.ray import ResourcePoolManager

        resource_pool_spec = {}
        mapping = {}

        uid = f"{name}_{int(time.time())}"
        training_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]
        if any(role in roles for role in training_roles):
            pool_name = f"{uid}_trainer_pool"
            trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
            resource_pool_spec[pool_name] = trainer_pool
            for role in training_roles:
                if role in roles:
                    mapping[role] = pool_name

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    def _create_rollouter(self, config):
        """Create and initialize CoTrainRollouter."""
        rollouter = CoTrainRollouter.remote(
            config=config,
            tokenizer=self.components["attacker_tokenizer"],
            processor=self.components["attacker_processor"],
            defender_tokenizer=self.components["defender_tokenizer"],
        )
        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())
        self.components["rollouter"] = rollouter
        logger.info("[CO-TRAIN] CoTrainRollouter created successfully")

    def _build_rollout_config(self, base_config, role: str):
        """Build a standalone rollout config for LLMServerManager.

        Each role gets its own LLMServerManager with independent GPU allocation.
        Roles: "attacker", "defender", "old_attacker", "old_defender"

        The attacker model is a text-only Qwen3.5 checkpoint.  Its Hugging
        Face ``qwen3_5_text`` config is newer than the vLLM config registry,
        while the vLLM Qwen3.5 implementation expects an outer
        ``Qwen3_5Config``.  Normalize the outer config and retain the full
        original text config so vLLM selects its CausalLM implementation
        without losing the checkpoint's long-context and hybrid-layer fields.
        """
        cfg = copy.deepcopy(base_config)
        cotrain_cfg = base_config.cotrain
        with open_dict(cfg):
            cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

            if role == "attacker":
                cfg.actor_rollout_ref.model.path = cotrain_cfg.attacker_model_path
                cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = cotrain_cfg.get(
                    "current_attacker_tp", 1
                )
                cfg.actor_rollout_ref.rollout.nnodes = cotrain_cfg.get("current_attacker_rollout_nnodes", 1)
                cfg.actor_rollout_ref.rollout.n_gpus_per_node = cotrain_cfg.get(
                    "current_attacker_rollout_gpus_per_node", 4
                )
                cfg.actor_rollout_ref.rollout.max_model_len = cotrain_cfg.get(
                    "current_attacker_max_model_len",
                    cotrain_cfg.get("attacker_max_prompt_length", 8192)
                    + cotrain_cfg.get("attacker_max_response_length", 4096)
                )
                cfg.actor_rollout_ref.rollout.hf_overrides = self._qwen35_text_vllm_overrides(
                    cotrain_cfg.attacker_model_path
                )
                cfg.actor_rollout_ref.rollout.engine_kwargs = {
                    "vllm": {
                        "skip_mm_profiling": True,
                        # Required for the request-scoped raw ChatML template
                        # used by the SFT-aligned adaptive attacker loop.
                        "trust_request_chat_template": True,
                    }
                }
                cfg.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs = {"nccl": {"group_name": "atk_ckpt"}}
                cfg.actor_rollout_ref.rollout.prometheus.served_model_name = None
            elif role == "defender":
                cfg.actor_rollout_ref.model.path = cotrain_cfg.defender_model_path
                cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = cotrain_cfg.get(
                    "current_defender_tp", 1
                )
                cfg.actor_rollout_ref.rollout.nnodes = cotrain_cfg.get("current_defender_rollout_nnodes", 1)
                cfg.actor_rollout_ref.rollout.n_gpus_per_node = cotrain_cfg.get(
                    "current_defender_rollout_gpus_per_node", 8
                )
                cfg.actor_rollout_ref.rollout.max_model_len = cotrain_cfg.get(
                    "current_defender_max_model_len", 40960
                )
                cfg.actor_rollout_ref.rollout.engine_kwargs = {
                    "vllm": {
                        "skip_mm_profiling": True,
                        # Defender requests use the reasoning-preserving
                        # ChatML template so historical <think> tokens remain
                        # aligned across multi-turn PPO traces.
                        "trust_request_chat_template": True,
                    }
                }
                cfg.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs = {"nccl": {"group_name": "def_ckpt"}}
                cfg.actor_rollout_ref.rollout.prometheus.served_model_name = None
            elif role == "old_attacker":
                cfg.actor_rollout_ref.model.path = cotrain_cfg.attacker_model_path
                cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = cotrain_cfg.get(
                    "old_attacker_tp", 2
                )
                cfg.actor_rollout_ref.rollout.nnodes = cotrain_cfg.get("old_attacker_rollout_nnodes", 1)
                cfg.actor_rollout_ref.rollout.n_gpus_per_node = cotrain_cfg.get(
                    "old_attacker_rollout_gpus_per_node", 4
                )
                cfg.actor_rollout_ref.rollout.max_model_len = cotrain_cfg.get(
                    "old_attacker_max_model_len", 12288
                )
                cfg.actor_rollout_ref.rollout.gpu_memory_utilization = cotrain_cfg.get(
                    "old_model_gpu_mem_util", 0.85
                )
                cfg.actor_rollout_ref.rollout.hf_overrides = self._qwen35_text_vllm_overrides(
                    cotrain_cfg.attacker_model_path
                )
                cfg.actor_rollout_ref.rollout.engine_kwargs = {
                    "vllm": {
                        "skip_mm_profiling": True,
                        "trust_request_chat_template": True,
                    }
                }
                cfg.actor_rollout_ref.rollout.prometheus.served_model_name = "old_attacker"
            elif role == "old_defender":
                cfg.actor_rollout_ref.model.path = cotrain_cfg.defender_model_path
                cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = cotrain_cfg.get(
                    "old_defender_tp", 2
                )
                cfg.actor_rollout_ref.rollout.nnodes = cotrain_cfg.get("old_defender_rollout_nnodes", 1)
                cfg.actor_rollout_ref.rollout.n_gpus_per_node = cotrain_cfg.get(
                    "old_defender_rollout_gpus_per_node", 4
                )
                cfg.actor_rollout_ref.rollout.max_model_len = cotrain_cfg.get(
                    "old_defender_max_model_len", 40960
                )
                cfg.actor_rollout_ref.rollout.gpu_memory_utilization = cotrain_cfg.get(
                    "old_model_gpu_mem_util", 0.85
                )
                cfg.actor_rollout_ref.rollout.engine_kwargs = {
                    "vllm": {
                        "skip_mm_profiling": True,
                        "trust_request_chat_template": True,
                    }
                }
                cfg.actor_rollout_ref.rollout.prometheus.served_model_name = "old_defender"

        return cfg

    @staticmethod
    def _qwen35_text_vllm_overrides(model_path: str) -> dict:
        """Adapt a pure-text Qwen3.5 config for vLLM's Qwen3.5 registry.

        vLLM accepts a ``model_type`` override before selecting its config
        class.  Its Qwen3.5 outer config owns a nested ``text_config``; using
        the source config there preserves non-default fields such as
        ``max_position_embeddings`` and the gated-delta layer schedule.
        """
        config_path = os.path.join(str(model_path), "config.json")
        try:
            with open(config_path, encoding="utf-8") as f:
                text_config = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load attacker config for vLLM overrides: {config_path}"
            ) from exc

        return {
            "model_type": "qwen3_5",
            # This alias is registered by vllm_async_server.  The stock vLLM
            # registry otherwise falls back to Transformers' multimodal
            # Qwen3.5 implementation for this architecture name.
            "architectures": ["Qwen3_5TextForCausalLM"],
            "text_config": text_config,
        }

    def _get_current_model_urls(self, role: str) -> list[str]:
        """Get HTTP URLs of current model vLLM instances managed by LLMServerManager."""
        mgr = self.components.get(f"{role}_llm_server_manager")
        if mgr:
            return [f"http://{addr}/v1" for addr in mgr.get_addresses()]
        return []

    def _run_training_loop(self):
        """Launch the rollouter and configured trainers concurrently.

        Weight sync happens automatically: each trainer calls _fit_update_weights()
        which pushes via NCCL to its replicas after every trigger_parameter_sync_step.
        """
        self.running = True

        logger.info(
            "[CO-TRAIN] Starting training loop (%s)...",
            "rollouter + 2 trainers",
        )
        rollouter_future = self.components["rollouter"].fit.remote()
        def_trainer_future = self.components["defender_trainer"].fit.remote()

        future_to_name = {
            rollouter_future: "rollouter",
            def_trainer_future: "defender_trainer",
        }
        atk_trainer_future = self.components["attacker_trainer"].fit.remote()
        future_to_name[atk_trainer_future] = "attacker_trainer"
        futures = list(future_to_name.keys())

        try:
            while futures:
                done_futures, remaining_futures = ray.wait(futures, num_returns=1, timeout=None)

                for future in done_futures:
                    name = future_to_name[future]
                    try:
                        ray.get(future)
                        logger.info(f"[CO-TRAIN] {name} completed successfully")
                    except Exception as e:
                        logger.error(f"[CO-TRAIN] {name} failed: {e}")
                        for remaining in remaining_futures:
                            ray.cancel(remaining)
                        raise

                futures = list(remaining_futures)

        except Exception as e:
            logger.error(f"[CO-TRAIN] Training failed: {e}")
            raise
        finally:
            self.running = False
            for mq_name in ["attacker_mq_client", "defender_mq_client"]:
                mq_client = self.components.get(mq_name)
                if mq_client:
                    asyncio.run(mq_client.clear_queue())

            logger.info("[CO-TRAIN] Training completed or interrupted")


# ==================== Hydra Entry Point ====================


@hydra.main(config_path="../config", config_name="cotrain_config", version_base=None)
def main(config):
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    from verl.trainer.main_ppo import run_ppo

    # Validate config
    if not hasattr(config, "cotrain"):
        raise RuntimeError("Must provide cotrain config section")
    if not hasattr(config, "async_training"):
        raise RuntimeError("Must provide async_training config section")

    auto_set_device(config)
    OmegaConf.resolve(config)

    # Set rollout config (compatibility with verl)
    with open_dict(config):
        config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
        config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    config = migrate_legacy_reward_impl(config)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(namespace="cotrain_self_play")

    # Launch orchestrator
    orchestrator = CoTrainOrchestrator.remote()
    try:
        ray.get(orchestrator.run.remote(config))
    except Exception as e:
        logger.error(f"[CO-TRAIN] Failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
