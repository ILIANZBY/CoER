"""Process-start hook for the pure-text Qwen3.5 vLLM architecture alias.

vLLM's V1 EngineCore launches worker Python processes independently from the
HTTP server.  This hook is imported by Python's standard ``site`` machinery
in each of those processes, so the registry alias is available before the
engine reconstructs its model configuration.
"""

import os


if os.getenv("VERL_QWEN35_TEXT_VLLM_REGISTRY") == "1":
    # vLLM inspects model classes in a fresh Python subprocess.  Patching the
    # API-server wrapper alone therefore does not cover that subprocess, while
    # the SciPy version in the training image still imports the removed NumPy
    # scalar aliases.  Keep the compatibility shim behind the same opt-in flag
    # as the local Qwen3.5 registry.
    import numpy as np

    np.long = np.int64
    np.ulong = np.uint64

    try:
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            Qwen3_5ForConditionalGenerationConfig,
        )
        from vllm.model_executor.models.registry import ModelRegistry
    except ImportError:
        # The hook is also present in lightweight setup processes where vLLM
        # is intentionally unavailable.
        pass
    else:
        if "Qwen3_5TextForCausalLM" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "Qwen3_5TextForCausalLM",
                "verl.models.qwen3_5_text_vllm:Qwen3_5TextForCausalLM",
            )
        # Preserve the upstream Qwen3.5 setting for its float32 GDN state.
        MODELS_CONFIG_MAP.setdefault(
            "Qwen3_5TextForCausalLM",
            Qwen3_5ForConditionalGenerationConfig,
        )
