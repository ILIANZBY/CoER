"""vLLM adapter for the pure-text Qwen3.5 checkpoint architecture.

vLLM's built-in Qwen3.5 CausalLM implementation contains the required
Gated-Delta layers, but is not marked as hybrid.  The marker makes vLLM size
and pad Mamba/GDN and full-attention cache pages consistently before EngineCore
allocates its KV cache.
"""

import torch

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)


class Qwen3_5TextForCausalLM(Qwen3_5ForCausalLM, IsHybrid):
    """Pure-text Qwen3.5 CausalLM with hybrid-cache and M-RoPE support."""

    # The pure-text checkpoint still uses Qwen3.5's interleaved M-RoPE
    # frequency layout.  Unlike the multimodal wrapper, every axis has the
    # same monotonic text position, so no vision-grid processing is needed.
    # Declaring this protocol member is required by vLLM's request scheduler
    # before it accepts a three-axis position tensor.
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Return Qwen3.5 text-only M-RoPE positions.

        For a sequence without image/video embeddings, Qwen's reference
        multimodal position builder broadcasts ``arange(seq_len)`` to all
        three M-RoPE axes.  This model intentionally accepts text only;
        failing explicitly keeps an accidental multimodal request from being
        assigned silently incorrect positions.
        """
        if mm_features:
            raise ValueError("Qwen3_5TextForCausalLM does not accept multimodal inputs")

        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.repeat(3, 1), 0

    # The upstream CausalLM class reuses the hybrid decoder layers but omits
    # the helpers exposed by its conditional-generation wrapper.  vLLM's
    # hybrid cache configurator needs these class methods before a model
    # instance exists, so delegate to that wrapper's architecture-equivalent
    # implementations.
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
