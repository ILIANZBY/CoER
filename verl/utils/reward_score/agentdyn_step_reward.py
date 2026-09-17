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
"""Step-wise reward function for AgentDyn PPO training.

This module provides a custom reward function that reads per-turn step-wise
rewards from the agent loop output and constructs per-token rm_scores.

When used with REINFORCE++ (adv_estimator=reinforce_plus_plus), this enables
proper credit assignment: rewards at turn boundaries propagate backwards via
discounted returns, giving earlier actions appropriate advantage signals.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | list[dict]:
    """Compute step-wise reward for AgentDyn PPO trajectory.

    This function serves dual purposes:
    1. Returns the scalar total reward (sum of step rewards) for standard compatibility
    2. Stores per-token reward positions in extra_info for downstream processing

    In the standard flow (reward placed at last token), the scalar value is what
    matters. For step-wise REINFORCE++, the per-token positions stored in
    extra_info["step_rm_scores"] are used by the custom postprocessing.

    Args:
        solution_str: The agent's final output string.
        ground_truth: Dict with task configuration.
        extra_info: Contains "step_rm_scores" list of (position, reward) tuples.

    Returns:
        Float reward score (sum of all step rewards).
    """
    if extra_info is None:
        extra_info = {}

    # If step-wise rewards already computed by the agent loop, use them
    step_rm_scores = extra_info.get("step_rm_scores", [])
    if step_rm_scores:
        return sum(r for _, r in step_rm_scores)

    # Fallback: use pre-computed total reward
    if "agentdyn_reward" in extra_info:
        return extra_info["agentdyn_reward"]

    # Last resort: no reward info available
    return 0.0
