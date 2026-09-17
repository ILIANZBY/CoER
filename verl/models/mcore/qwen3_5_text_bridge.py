"""MBridge support for Hugging Face Qwen3.5 text-only checkpoints.

The upstream MBridge package currently registers the multimodal ``qwen3_5``
configuration only.  Qwen3.5 text checkpoints use ``qwen3_5_text`` and have
the same gated-delta transformer, but do not contain a vision configuration or
the ``language_model`` parameter prefix used by the multimodal bridge.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from mbridge.core import Bridge, register_model
from mbridge.core.safetensor_io import SafeTensorIO
from mbridge.models.qwen3_5.base_bridge import Qwen3_5VlBaseBridge
from mbridge.models.qwen3_5.model import Qwen3_5GPTModel
from mbridge.models.qwen3_5.transformer_config import Qwen3_5VLTransformerConfig


def _gdn_local_head_parameter(module, parameter: torch.Tensor, local_parameter: torch.Tensor, alpha: torch.Tensor):
    """Return a GatedDeltaNet parameter aligned with ``alpha``'s local heads.

    Qwen3.5 text checkpoints reach MBridge with ``dt_bias``/``A_log`` still
    replicated over TP on some worker roles.  Megatron's GatedDeltaNet then
    CP-slices those replicated tensors in the wrong order: the activation is
    TP+CP local while the two head parameters retain an extra TP shard.  The
    resulting broadcast failure first appears in reference-policy log-prob.

    Prefer Megatron's normal CP-local argument whenever it already matches.
    Otherwise derive the local shard from the original parameter in TP-major,
    CP-minor order, which is the same order used when GatedDeltaNet constructs
    its per-TP head parameters.
    """
    local_heads = int(alpha.shape[-1])
    if local_parameter.numel() == local_heads:
        return local_parameter.reshape(-1)

    flat = parameter.reshape(-1)
    if flat.numel() == local_heads:
        return flat

    cp_group = module.pg_collection.cp
    tp_group = module.pg_collection.tp
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    tp_size, tp_rank = tp_group.size(), tp_group.rank()

    # A TP-local parameter needs only its CP shard. A replicated parameter
    # needs both the TP and CP shard. Keep a modulo fallback for future model
    # variants that use a different but evenly divisible head layout.
    shard_count = flat.numel() // local_heads
    if flat.numel() % local_heads:
        raise RuntimeError(
            "Qwen3.5 GatedDeltaNet parameter cannot be aligned with local heads: "
            f"parameter={flat.numel()}, local_heads={local_heads}"
        )
    if shard_count == cp_size:
        shard_idx = cp_rank
    elif shard_count == tp_size * cp_size:
        shard_idx = tp_rank * cp_size + cp_rank
    else:
        shard_idx = (tp_rank * cp_size + cp_rank) % shard_count
    return flat.narrow(0, shard_idx * local_heads, local_heads)


def _patch_qwen35_text_gated_delta_net() -> None:
    """Fix text-only Qwen3.5 head-parameter sharding in installed MCore."""
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if getattr(GatedDeltaNet, "_verl_qwen35_text_head_shard_fix", False):
        return

    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        A_log_local_cp = _gdn_local_head_parameter(self, self.A_log, A_log_local_cp, alpha)
        dt_bias_local_cp = _gdn_local_head_parameter(self, self.dt_bias, dt_bias_local_cp, alpha)
        g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
        return g, beta.sigmoid()

    # Replace MCore's ``@jit_fuser`` method after the upstream module is
    # imported.  The eager expression retains its CUDA kernels while avoiding
    # the stale compiled shape specialization for these text-only parameters.
    GatedDeltaNet._compute_g_and_beta = _compute_g_and_beta
    GatedDeltaNet._verl_qwen35_text_head_shard_fix = True


_patch_qwen35_text_gated_delta_net()


def _patch_qwen35_text_mrope_position_ids() -> None:
    """Allow the Qwen3.5 MRoPE helper to consume ordinary text position IDs.

    MBridge reuses the Qwen3.5-VL GPT implementation for the text-only
    checkpoint.  That implementation correctly uses MRoPE, but its rotary
    helper assumes a visual ``[3, batch, sequence]`` position tensor.  PPO's
    packed text batches provide the normal ``[batch, sequence]`` tensor
    instead, which only becomes visible when the attacker reaches its critic
    forward pass.  Pure text is the degenerate MRoPE case: time, height, and
    width positions are identical.  Expanding the two-dimensional IDs across
    the three MRoPE axes preserves the standard text RoPE frequencies while
    leaving genuine multimodal 3-D inputs untouched.
    """
    from mbridge.models.qwen3_vl.rope_utils import Qwen3VLMultimodalRotaryEmbedding

    if getattr(Qwen3VLMultimodalRotaryEmbedding, "_verl_text_position_ids_fix", False):
        return

    original_forward = Qwen3VLMultimodalRotaryEmbedding.forward

    def forward(self, position_ids: torch.Tensor, mrope_section, **kwargs):
        if position_ids is not None and position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return original_forward(self, position_ids, mrope_section, **kwargs)

    Qwen3VLMultimodalRotaryEmbedding.forward = forward
    Qwen3VLMultimodalRotaryEmbedding._verl_text_position_ids_fix = True


_patch_qwen35_text_mrope_position_ids()


def _qwen35_text_packed_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """Safe packed-sequence GatedDeltaNet rule for Qwen3.5 text checkpoints.

    The FLA Triton path used by the installed MCore can issue an illegal memory
    access for this checkpoint's TP/CP-local head layout once a packed PPO
    trajectory reaches the reference-policy pass.  MCore's deterministic
    torch implementation is correct, but only accepts one un-packed sequence.
    Split the packed batch at its CUDA ``cu_seqlens`` boundaries and run that
    same GPU implementation per sequence.  This keeps all model computation
    on GPU while avoiding the unsafe Triton kernel.
    """
    from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule

    # Qwen3.5 uses grouped linear attention: query/key are expanded from the
    # smaller key-head count to the larger value-head count in MCore, whereas
    # FLA accepts the still-grouped ``g``/``beta`` tensors.  The deterministic
    # MCore fallback is deliberately simpler and requires every input to have
    # the value-head count.  Mirror MCore's query/key expansion here before
    # invoking it (e.g. 8 control heads -> 16 value heads).
    value_heads = int(value.shape[2])

    def _expand_control_heads(name: str, control: torch.Tensor) -> torch.Tensor:
        control_heads = int(control.shape[2])
        if control_heads == value_heads:
            return control
        if value_heads % control_heads:
            raise RuntimeError(
                f"GatedDeltaNet {name} heads cannot align with value heads: "
                f"{control_heads} -> {value_heads}"
            )
        return control.repeat_interleave(value_heads // control_heads, dim=2).contiguous()

    g = _expand_control_heads("g", g)
    beta = _expand_control_heads("beta", beta)

    if cu_seqlens is None:
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    # Packed GatedDeltaNet uses a single batch dimension and resets its
    # recurrent state at every sequence boundary.
    if query.shape[0] != 1:
        raise RuntimeError(f"Expected packed GatedDeltaNet batch=1, got {query.shape[0]}")
    offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    total_tokens = query.shape[1]
    outputs = []
    last_state = None
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if start >= total_tokens:
            break
        end = min(end, total_tokens)
        if end <= start:
            continue
        output, last_state = torch_chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            chunk_size=chunk_size,
            initial_state=None,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        outputs.append(output)
    if not outputs:
        raise RuntimeError("Packed GatedDeltaNet received no non-empty sequences")
    return torch.cat(outputs, dim=1), last_state if output_final_state else None


def _install_qwen35_text_gated_delta_rule(model: torch.nn.Module) -> None:
    """Use the packed-safe rule only for the local text-only Qwen3.5 model."""
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            module.gated_delta_rule = _qwen35_text_packed_gated_delta_rule


class _PyTorchBinIO(SafeTensorIO):
    """Lazy reader for the single-file legacy Hugging Face checkpoint format."""

    def __init__(self, hf_dir: str):
        self.hf_dir = hf_dir
        self.index = {}
        self.origin_index = {}
        self._path = os.path.join(hf_dir, "pytorch_model.bin")
        self._state_dict = None

    def _weights(self) -> dict[str, torch.Tensor]:
        if self._state_dict is None:
            self._state_dict = torch.load(
                self._path, map_location="cpu", mmap=True, weights_only=True
            )
        return self._state_dict

    def load_some_hf_weight(self, hf_weight_names: list[str]) -> dict[str, torch.Tensor]:
        weights = self._weights()
        missing = set(hf_weight_names).difference(weights)
        if missing:
            raise ValueError(f"Weights {missing} not found in {self._path}")
        return {name: weights[name] for name in hf_weight_names}

    def load_hf_weight_names(self) -> list[str]:
        return list(self._weights())


@register_model("qwen3_5_text")
class Qwen3_5TextBridge(Qwen3_5VlBaseBridge):
    """Text-only Qwen3.5 bridge with Qwen3.5 linear-attention support."""

    TransformerConfigClass = Qwen3_5VLTransformerConfig

    _CONFIG_MAPPING = {
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_query_groups": "num_key_value_heads",
        "ffn_hidden_size": "intermediate_size",
        "attention_dropout": "attention_dropout",
        "layernorm_epsilon": "rms_norm_eps",
        "hidden_dropout": ("hidden_dropout", 0.0),
        "kv_channels": ("head_dim", None),
    }

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight",
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "self_attention.dt_bias": [
            "model.layers.{layer_number}.linear_attn.dt_bias",
        ],
        "self_attention.A_log": [
            "model.layers.{layer_number}.linear_attn.A_log",
        ],
        "self_attention.in_proj.weight": [
            "model.layers.{layer_number}.linear_attn.in_proj_qkv.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_z.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_b.weight",
            "model.layers.{layer_number}.linear_attn.in_proj_a.weight",
        ],
        "self_attention.conv1d.weight": [
            "model.layers.{layer_number}.linear_attn.conv1d.weight",
        ],
        "self_attention.out_norm.weight": [
            "model.layers.{layer_number}.linear_attn.norm.weight",
        ],
        "self_attention.out_proj.weight": [
            "model.layers.{layer_number}.linear_attn.out_proj.weight",
        ],
        "self_attention.in_proj.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight",
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "mlp.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.down_proj.weight",
        ],
    }
    _OTHER_MAPPING = {}

    def __init__(self, hf_config, *args, **kwargs):
        # The shared Qwen3.5 conversion helpers consume ``text_config``.  A
        # shallow copy provides that view without changing the text checkpoint
        # into a multimodal model or introducing a self reference.
        if not hasattr(hf_config, "text_config"):
            hf_config.text_config = copy.copy(hf_config)
        super().__init__(hf_config, *args, **kwargs)

    def _build_config(self):
        rope = self.hf_config.rope_scaling
        config = self._build_base_config(
            layernorm_epsilon=self.hf_config.rms_norm_eps,
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
            distribute_saved_activations=False,
            cp_comm_type="p2p",
            qk_layernorm=True,
            layernorm_zero_centered_gamma=True,
            attention_output_gate=True,
            kv_channels=self.hf_config.head_dim,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=self.hf_config.full_attention_interval,
            linear_conv_kernel_dim=self.hf_config.linear_conv_kernel_dim,
            linear_key_head_dim=self.hf_config.linear_key_head_dim,
            linear_value_head_dim=self.hf_config.linear_value_head_dim,
            linear_num_key_heads=self.hf_config.linear_num_key_heads,
            linear_num_value_heads=self.hf_config.linear_num_value_heads,
            rotary_percent=rope.get("partial_rotary_factor", 0.25),
            mrope_section=rope.get("mrope_section", [11, 11, 10]),
            apply_rotary_pos_emb_in_fp32=True,
        )
        # MBridge enables sequence parallel automatically whenever TP > 1.
        # The installed GatedDeltaNet CP<->HP conversion does not preserve a
        # packed text trajectory under that layout: its local attention output
        # contains half the tokens while the residual remains full length.
        # Keeping TP=2 but disabling sequence parallel makes the packed token
        # layout consistent through the text-only GatedDeltaNet path.
        config.sequence_parallel = False
        return config

    def _model_provider(
        self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]
    ):
        share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        def provider(pre_process, post_process, vp_stage: Optional[int] = None):
            transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
            vocab_size = self.hf_config.vocab_size
            self.vocab_size = vocab_size
            self.padded_vocab_size = vocab_size
            if self.make_vocab_size_divisible_by is not None:
                self.padded_vocab_size = int(
                    math.ceil(vocab_size / self.make_vocab_size_divisible_by)
                    * self.make_vocab_size_divisible_by
                )

            model = Qwen3_5GPTModel(
                config=self.config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=self.padded_vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type="mrope",
                rotary_percent=self.hf_config.rope_scaling.get("partial_rotary_factor", 0.25),
                rotary_base=self.hf_config.rope_scaling.get("rope_theta", 10000000),
                mtp_block_spec=None,
                scatter_embedding_sequence_parallel=False,
            )
            _install_qwen35_text_gated_delta_rule(model)
            for callback in post_model_creation_callbacks:
                callback(
                    model,
                    pre_process=pre_process,
                    post_process=post_process,
                    config=self.config,
                    hf_config=self.hf_config,
                )
            return model

        return provider

    def _get_safetensor_io(self, weights_path: str):
        if os.path.isfile(os.path.join(weights_path, "pytorch_model.bin")) and not any(
            name.endswith(".safetensors") for name in os.listdir(weights_path)
        ):
            return _PyTorchBinIO(weights_path)
        return super()._get_safetensor_io(weights_path)

    # Qwen3_5VlBaseBridge uses a ``language_model.decoder`` prefix.  Text-only
    # GPTModel uses the standard ``decoder`` prefix, for which Bridge already
    # provides the correct layer-number handling.
    _weight_name_mapping_mcore_local_to_global = Bridge._weight_name_mapping_mcore_local_to_global
    _weight_name_mapping_attention = Bridge._weight_name_mapping_attention
    _weight_name_mapping_mlp = Bridge._weight_name_mapping_mlp
    _weight_name_mapping_other = Bridge._weight_name_mapping_other
    _weight_name_mapping_mcore_to_hf = Bridge._weight_name_mapping_mcore_to_hf
