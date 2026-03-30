# env.py
"""
KubeCost-Gym Environment (OpenEnv Interface).

Stub phase: method signatures with complete docstrings.
Each method body is placeholder (pass).

Reference: PROJECT_SPEC.md §2 OpenEnv Interface, §4 Environment Interface
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any
from models import Observation, Action, EnvState, ActionType, TrajectoryStep


class KubeCostEnv:
    """Kubernetes cost optimization environment.
    
    Implements OpenEnv interface for RL agents:
      - reset() → Observation
      - step(action) → (Observation, float, bool, dict)
      - state() → EnvState (typed, not dict)
    
    Uses deterministic pre-recorded traces for reproducibility.
    """
    
    def __init__(self, trace_path: str):
        """
        Initialize environment from deterministic trace.
        
        Args:
            trace_path: Path to JSON trace file
                       (e.g., 'traces/trace_v1_coldstart.json')
        
        Implementation plan:
            - Load JSON trace from trace_path
            - Validate trace schema (task_name, task_difficulty, steps[])
            - Initialize step counter to 0
            - Store reference to trace data
            - Initialize episode state (replicas, node_size, etc.)
        
        Raises:
            FileNotFoundError: If trace_path not found
            ValueError: If trace schema invalid
        """
        self.trace_path = Path(trace_path)
        # STUB: Load JSON trace, validate, initialize
        pass
    
    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        
        Returns:
            Observation: Typed Pydantic model instance (NOT dict)
        
        Validation (spec §2):
            - Returns Pydantic Observation instance
            - All fields within declared ranges (Field(ge=..., le=...))
            - 10-element node_bin_density vector
        
        Common Failure (spec §2):
            ✗ Returns dict instead of Observation instance
            ✓ Should return: Observation(cpu_usage_pct=..., mem_usage_pct=..., ...)
        
        Implementation plan:
            - Reset step counter to 0
            - Load first step from trace
            - Parse observation fields from trace[step]["observation"]
            - Return Observation(**obs_dict)
        """
        # STUB: Return initial Observation
        pass
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step of environment dynamics.
        
        Args:
            action: Action model with action_type field
        
        Returns:
            4-tuple: (Observation, float, bool, dict)
                - Observation: New state after action
                - float: Reward signal this step
                - bool: Episode done flag
                - dict: Info/metadata
        
        Validation (spec §2):
            - 4-tuple return (not None, not 3-tuple)
            - observation is Observation instance (not dict)
            - reward is float (not int)
            - done is bool
            - info is dict
        
        Common Failure (spec §2):
            ✗ Returns None (stub body)
            ✗ Returns 3-tuple: (obs, reward, done)
            ✗ reward is int: 1 (should be 1.0)
        
        Implementation plan:
            - Call _apply_action(action) to update state
            - Advance step counter
            - Load next observation from trace
            - Call _calculate_reward() for reward signal
            - Determine done: done = (step >= len(trace)) or episode_complete
            - Return 4-tuple
        """
        # STUB: Return (Observation, float, bool, dict)
        pass
    
    def state(self) -> EnvState:
        """
        Get current environment state.
        
        Returns:
            EnvState: Typed Pydantic model instance (NOT dict)
        
        Validation (spec §2):
            - Returns Pydantic EnvState instance (not bare dict)
            - Non-null and serializable
            - All fields: step, replicas, node_size, prev_steal_pct
        
        Common Failure (spec §2):
            ✗ Returns dict: {"step": 5, "replicas": 3, ...}
            ✓ Should return: EnvState(step=5, replicas=3, node_size=NodeSizeClass.MEDIUM, ...)
        
        Implementation plan:
            - Gather current state variables from self
            - Return EnvState(step=self.step, replicas=..., node_size=..., prev_steal_pct=...)
        """
        # STUB: Return EnvState instance
        pass
    
    def _apply_action(self, action: Action) -> None:
        """
        Apply action to environment dynamics (internal helper).
        
        Called by step() before reward calculation.
        
        Implementation plan:
            - Parse action.action_type (ActionType enum)
            - Branch on action type:
              * SCALE_REPLICAS(±N): Update self.replicas
              * UPGRADE_NODE: Update self.node_size (irreversible for 1 step)
              * REBALANCE_NODE: Trigger rebalancing logic
              * MAINTAIN: No state change
            - Update internal state variables
            - Store action for trajectory logging
        
        Reference: PROJECT_SPEC.md §4 Action Space
        """
        # STUB: Update self state based on action
        pass
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward signal this step (internal helper).
        
        Returns:
            float: Reward in range [R_min, R_max]
        
        Constraints (spec §3.3 Reward Specification):
            - R_min per step: -20.0 (SLA breach penalty)
            - R_max per step: +10.5 (perfect uptime + proactive bonus)
            - CRITICAL: No sparse reward cliffs
            - Linear ramp penalty for p99 ∈ [200, 300ms)
        
        Audit Fix 03 (spec §5):
            ✗ Binary reward cliff at p99=300ms
            ✓ Linear ramp penalty for warning zone [200, 300ms)
        
        Formula (draft, specify in Phase 3):
            R = (10.0 × Uptime) − (5.0 × Cost/Budget) − RampPenalty(p99)
              + ProactiveBonus(steal_drop, p99_good)
            Where:
              Uptime = 1.0 if p99 < 300ms, else 0.0
              RampPenalty(p99) = (p99−200)/100 × 5.0 when p99 ∈ [200, 300)
              SLABreach = −20.0 when p99 ≥ 300ms
              ProactiveBonus = +0.5 when steal_drops AND p99 < 300ms
        
        Implementation plan:
            - Get current observation
            - Check p99_latency_ms for gradients
            - Check cpu_steal_pct for proactive bonus
            - Compute cost penalty
            - Sum components into single float
            - Clamp to [R_min, R_max]
            - Return float reward
        """
        # STUB: Compute and return float reward
        pass
    
    def _load_trace(self, trace_path: Path) -> dict:
        """
        Load and validate deterministic trace JSON (internal helper).
        
        Args:
            trace_path: Path to JSON file
        
        Returns:
            dict: Validated trace structure
        
        Expected schema:
            {
              "task_name": "cold_start|efficient_squeeze|entropy_storm",
              "task_difficulty": "easy|medium|hard",
              "steps": [
                {
                  "step": 0,
                  "observation": {...},
                  "action": "MAINTAIN",
                  "dynamics": {...}
                },
                ...
              ]
            }
        
        Implementation plan:
            - Load JSON from trace_path
            - Validate required keys: task_name, task_difficulty, steps
            - Validate step structure (each has observation, action, dynamics)
            - Return parsed dict
        """
        # STUB: Load and validate JSON trace
        pass
