"""
AgentDyn Co-Training: Attacker-Defender Self-Play System

Architecture:
  - CoTrainOrchestrator: Top-level coordinator managing all components
  - CoTrainRollouter: Unified rollouter with 4 model groups + dual-queue routing
  - CoTrainAgentLoop: Unified agent loop (attacker gen + defender eval)
  - OldModelManager: Manages periodic refresh of old model vLLM instances
"""
