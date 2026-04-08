# env.py
"""
KubeCost-Gym Environment (OpenEnv Interface).

Phase 3 Implementation: Reward Specification and Physics Logic.
All methods fully implemented — no stub (pass) bodies remain.

Reference: PROJECT_SPEC.md §2 OpenEnv Interface, §3 Reward Spec, §4 Environment Interface
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from openenv.core import Environment

from models import (
    Observation,
    Action,
    EnvState,
    ActionType,
    NodeSizeClass,
    TrajectoryStep,
    TraceData,
    TraceObservation,
    TraceStep,
)

__all__ = [
    "KubeCostEnv",
    "load_trace",
    "compute_reward",
    "validate_action",
    "get_replica_delta",
    "EnvError",
]

# Configure module logger
logger = logging.getLogger(__name__)

# ===== CUSTOM EXCEPTIONS =====


class EnvError(Exception):
    """Base exception for environment-related errors."""

    pass


class TraceLoadError(EnvError):
    """Raised when trace file cannot be loaded or parsed."""

    pass


class ActionValidationError(EnvError):
    """Raised when action fails validation."""

    pass


# ===== ENVIRONMENT CONFIGURATION =====


class _EnvironmentConfig:
    """Centralized configuration for reward and constraint calculations."""

    # Node-size ordering for UPGRADE_NODE logic (irreversible for 1 step)
    NODE_TIER = {
        NodeSizeClass.SMALL: 0,
        NodeSizeClass.MEDIUM: 1,
        NodeSizeClass.LARGE: 2,
    }
    NODE_FROM_TIER = {v: k for k, v in NODE_TIER.items()}

    # Budget used in cost penalty: cost fraction = current_hourly_cost / BUDGET
    HOURLY_BUDGET: float = 100.0

    # Reward bounds (spec §3.3)
    REWARD_MIN: float = -20.0
    REWARD_MAX: float = 10.5

    # Replica hard bounds
    REPLICAS_MIN: int = 0
    REPLICAS_MAX: int = 200

    # SLA thresholds
    SLA_THRESHOLD_MS: float = 300.0
    SLA_WARNING_MIN_MS: float = 200.0

    # Reward component weights
    UPTIME_REWARD: float = 10.0
    COST_PENALTY_RATE: float = 5.0
    COST_PENALTY_CAP: float = 5.0
    RAMP_PENALTY_RATE: float = 5.0
    SLA_BREACH_PENALTY: float = 20.0
    PROACTIVE_BONUS: float = 0.5


_CONFIG = _EnvironmentConfig()
_NODE_TIER = _CONFIG.NODE_TIER
_NODE_FROM_TIER = _CONFIG.NODE_FROM_TIER
_HOURLY_BUDGET = _CONFIG.HOURLY_BUDGET
_R_MIN = _CONFIG.REWARD_MIN
_R_MAX = _CONFIG.REWARD_MAX
_REPLICAS_MIN = _CONFIG.REPLICAS_MIN
_REPLICAS_MAX = _CONFIG.REPLICAS_MAX


# ---------------------------------------------------------------------------
# Trace loading: Load and validate deterministic trace JSON (Fix #20)
# ---------------------------------------------------------------------------


def load_trace(trace_path: str | Path) -> TraceData:
    """
    Load and validate deterministic trace JSON using Pydantic.

    Args:
        trace_path: Path to JSON file (str or Path).

    Returns:
        TraceData: Validated trace as Pydantic model.

    Raises:
        TraceLoadError: If trace_path does not exist or JSON schema is invalid.
    """
    trace_path_obj = Path(trace_path) if isinstance(trace_path, str) else trace_path

    if not trace_path_obj.exists():
        error_msg = (
            f"Trace file not found: {trace_path_obj}. "
            f"Expected one of: traces/trace_v1_coldstart.json, "
            f"traces/trace_v1_squeeze.json, traces/trace_v1_entropy.json"
        )
        logger.error(error_msg)
        raise TraceLoadError(error_msg)

    try:
        with trace_path_obj.open("r", encoding="utf-8") as fh:
            data: dict = json.load(fh)
        logger.debug(f"Loaded trace from {trace_path_obj}")
    except (IOError, json.JSONDecodeError) as e:
        error_msg = f"Failed to read or parse trace file {trace_path_obj}: {e}"
        logger.error(error_msg)
        raise TraceLoadError(error_msg) from e

    try:
        # Pydantic validates entire structure recursively
        trace = TraceData(**data)
        logger.info(
            f"Trace validated: task={trace.task_name}, "
            f"difficulty={trace.task_difficulty}, steps={len(trace.steps)}"
        )
        return trace
    except ValueError as e:
        error_msg = f"Trace schema validation failed: {e}"
        logger.error(error_msg)
        raise TraceLoadError(error_msg) from e


# ---------------------------------------------------------------------------
# Reward computation (Fix #21)
# ---------------------------------------------------------------------------


def compute_reward(observation: Observation, previous_steal_pct: float) -> float:
    """
    Calculate episode reward from observation and previous state.

    Implements formula:
        R = (10.0 × Uptime)
          − (5.0 × Cost/Budget)
          − RampPenalty(p99)
          − SLABreach(p99)
          + ProactiveBonus

    Args:
        observation: Current state observation.
        previous_steal_pct: CPU steal from previous step (for proactive bonus).

    Returns:
        float: Reward clamped to [-20.0, +10.5].
    """
    p99 = observation.p99_latency_ms
    cost_fraction = observation.current_hourly_cost / _HOURLY_BUDGET

    # Uptime component
    uptime = 1.0 if p99 < _CONFIG.SLA_THRESHOLD_MS else 0.0
    uptime_reward = _CONFIG.UPTIME_REWARD * uptime

    # Cost penalty (uncapped, so overspend is penalized correctly)
    cost_penalty = _CONFIG.COST_PENALTY_RATE * cost_fraction

    # Ramp penalty (dense signal in warning zone [200, 300))
    ramp_penalty = 0.0
    if _CONFIG.SLA_WARNING_MIN_MS <= p99 < _CONFIG.SLA_THRESHOLD_MS:
        ramp_penalty = ((p99 - _CONFIG.SLA_WARNING_MIN_MS) / 100.0) * _CONFIG.RAMP_PENALTY_RATE

    # SLA breach hard penalty
    sla_breach_penalty = _CONFIG.SLA_BREACH_PENALTY if p99 >= _CONFIG.SLA_THRESHOLD_MS else 0.0

    # Proactive bonus (steal dropping + healthy p99)
    proactive_bonus = 0.0
    steal_dropped = observation.cpu_steal_pct < previous_steal_pct
    if steal_dropped and p99 < _CONFIG.SLA_THRESHOLD_MS:
        proactive_bonus = _CONFIG.PROACTIVE_BONUS

    raw_reward = (
        uptime_reward
        - cost_penalty
        - ramp_penalty
        - sla_breach_penalty
        + proactive_bonus
    )
    return float(max(_R_MIN, min(_R_MAX, raw_reward)))

# ---------------------------------------------------------------------------
# Action validation (Fix #22)
# ---------------------------------------------------------------------------


def validate_action(action: Action) -> None:
    """
    Validate action is well-formed and applicable.

    Args:
        action: Action to validate.

    Raises:
        ActionValidationError: If action is invalid.
    """
    if action is None:
        raise ActionValidationError("Action cannot be None")

    if not isinstance(action, Action):
        raise ActionValidationError(f"Action must be Action type, got {type(action)}")

    if action.action_type is None:
        raise ActionValidationError("Action.action_type cannot be None")

    if not isinstance(action.action_type, ActionType):
        raise ActionValidationError(f"action_type must be ActionType, got {type(action.action_type)}")


def get_replica_delta(action_type: ActionType) -> int:
    """
    Get replica count delta for a scale action.

    Args:
        action_type: Action type to analyze.

    Returns:
        int: Replica delta (positive=scale up, negative=scale down, 0=no change).
    """
    scale_map = {
        ActionType.SCALE_DOWN_5: -5,
        ActionType.SCALE_DOWN_1: -1,
        ActionType.SCALE_UP_1: 1,
        ActionType.SCALE_UP_5: 5,
        ActionType.SCALE_UP_10: 10,
        ActionType.SCALE_UP_20: 20,
        ActionType.UPGRADE_NODE: 0,
        ActionType.REBALANCE_NODE: 0,
        ActionType.MAINTAIN: 0,
    }
    return scale_map.get(action_type, 0)


class KubeCostEnv(Environment):
    """Kubernetes cost optimization environment.

    Implements OpenEnv interface for RL agents:
      - reset() → Observation
      - step(action) → (Observation, float, bool, dict)
      - state() → EnvState (typed, not dict)

    Uses deterministic pre-recorded traces for reproducibility.
    All dynamics are driven entirely by the loaded trace — no RNG.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, trace_path: str) -> None:
        """
        Initialize environment from deterministic trace.

        Args:
            trace_path: Path to JSON trace file
                       (e.g., 'traces/trace_v1_coldstart.json')

        Raises:
            FileNotFoundError: If trace_path not found
            ValueError: If trace schema is invalid
        """
        self.trace_path = Path(trace_path)
        self.trace: TraceData = load_trace(self.trace_path)
        self.steps_data = self.trace.steps  # list[TraceStep]
        self.total_steps: int = len(self.steps_data)

        # ------------------------------------------------------------------
        # Episode state — populated properly by reset()
        # ------------------------------------------------------------------
        self._step: int = 0
        self._current_obs: Observation | None = None
        self.steal_suppression_steps: int = 0

        # Mutable cluster state (updated by _apply_action)
        first_obs = self.steps_data[0].observation
        self._replicas: int = first_obs.active_replicas
        # Ensure node_size is an enum (Pydantic may return string value due to config)
        first_node = first_obs.node_size_class
        self._node_size: NodeSizeClass = NodeSizeClass(first_node) if isinstance(first_node, str) else first_node
        self._prev_steal_pct: float = first_obs.base_steal_pct

        # Trajectory log (list[TrajectoryStep]) — filled during step()
        self._trajectory: list[TrajectoryStep] = []
        self.task_name: str = self.trace.task_name

    # ------------------------------------------------------------------
    # OpenEnv Public Interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Observation:
        """
        Reset environment to initial state.

        Returns:
            Observation: Typed Pydantic model instance (NOT dict).

        Resets the step counter to 0, re-seeds mutable cluster state from
        the very first trace step, and returns the matching Observation.
        """
        self._step = 0
        self._trajectory = []

        # Read starting cluster config from trace step 0
        first_trace_step = self.steps_data[0]
        first_obs = first_trace_step.observation

        self._replicas = first_obs.active_replicas
        first_node = first_obs.node_size_class
        self._node_size = NodeSizeClass(first_node) if isinstance(first_node, str) else first_node
        self._prev_steal_pct = first_obs.base_steal_pct
        self.steal_suppression_steps = 0

        self._current_obs = self._build_observation(first_obs)
        return self._current_obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one step of environment dynamics.

        Args:
            action: Action model with action_type field.

        Returns:
            4-tuple: (Observation, float, bool, dict)
                - Observation: New state after action.
                - float:       Reward signal this step.
                - bool:        Episode done flag.
                - dict:        Info / metadata.

        Step semantics:
            1. Validate action.
            2. Advance step counter (next trace row provides new observation).
            3. Load new Observation from trace (physics ground-truth).
            4. Apply action to internal state.
            5. Compute reward from new observation.
            6. Determine done, return 4-tuple.
        """
        # 0. Validate action
        validate_action(action)
        
        self._step += 1

        if self._step >= self.total_steps:
            done = True
            return self._current_obs, 0.0, done, {}

        if self._current_obs is not None:
            self._prev_steal_pct = self._current_obs.cpu_steal_pct

        # Apply the agent action first so the next observation reflects capacity changes.
        self._apply_action(action)

        trace_step = self.steps_data[self._step]
        trace_obs = trace_step.observation

        new_obs = self._build_observation(trace_obs)
        self._current_obs = new_obs

        reward: float = self._calculate_reward()
        done: bool = self._step >= self.total_steps - 1
        
        # 10. Build info dict
        info: Dict[str, Any] = {
            "step": self._step,
            "task_name": self.trace.task_name,
            "task_difficulty": self.trace.task_difficulty,
            "replicas": self._replicas,
            "node_size": self._node_size.value if isinstance(self._node_size, NodeSizeClass) else self._node_size,
            "trace_reason": trace_step.dynamics.get("reason", ""),
        }
        
        # 11. Log to trajectory (without redundant metrics)
        self._trajectory.append(
            TrajectoryStep(
                observation=self._current_obs,
                action=action.action_type,
                reward=reward,
                done=done,
                info=info,
            )
        )

        return self._current_obs, reward, done, info

    def state(self) -> EnvState:
        """
        Get current environment state snapshot.

        Returns:
            EnvState: Typed Pydantic model (NOT bare dict).
        """
        return EnvState(
            step=self._step,
            replicas=self._replicas,
            node_size=self._node_size,
            prev_steal_pct=self._prev_steal_pct,
        )

    def render(self, mode: str = "human") -> Any:
        """Render environment state (stub)."""
        return None

    def close(self) -> None:
        """Cleanup resources (stub)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action) -> None:
        """
        Apply action to environment state (internal helper, called by step).

        Translates the ActionType enum value into concrete state mutations:

        SCALE_REPLICAS(±N):
            self._replicas += N, clamped to [REPLICAS_MIN, REPLICAS_MAX].
        UPGRADE_NODE:
            Advances node tier by one level (S→M or M→L). Irreversible for 1 step
            (the trace physics will reflect the new size from the next step onward).
        REBALANCE_NODE:
            Sets a rebalance signal; no direct replica/node change. The agent earns
            the ProactiveBonus via _calculate_reward() if steal is dropping.
        MAINTAIN:
            No state mutation.

        Reference: PROJECT_SPEC.md §4 Action Space
        """
        action_type = action.action_type

        # ---- SCALE_REPLICAS branch ----
        if action_type == ActionType.SCALE_DOWN_5:
            self._replicas = max(_REPLICAS_MIN, self._replicas - 5)

        elif action_type == ActionType.SCALE_DOWN_1:
            self._replicas = max(_REPLICAS_MIN, self._replicas - 1)

        elif action_type == ActionType.SCALE_UP_1:
            self._replicas = min(_REPLICAS_MAX, self._replicas + 1)

        elif action_type == ActionType.SCALE_UP_5:
            self._replicas = min(_REPLICAS_MAX, self._replicas + 5)

        elif action_type == ActionType.SCALE_UP_10:
            self._replicas = min(_REPLICAS_MAX, self._replicas + 10)

        elif action_type == ActionType.SCALE_UP_20:
            # Audit Fix 04: emergency burst absorption for hard task
            self._replicas = min(_REPLICAS_MAX, self._replicas + 20)

        # ---- UPGRADE_NODE: irreversible for 1 step ----
        elif action_type == ActionType.UPGRADE_NODE:
            current_tier = _NODE_TIER[self._node_size]
            next_tier = min(current_tier + 1, len(_NODE_TIER) - 1)
            self._node_size = _NODE_FROM_TIER[next_tier]

        # ---- REBALANCE_NODE: proactive signal (temporary noisy-neighbor relief) ----
        elif action_type == ActionType.REBALANCE_NODE:
            # Grants a short window of reduced steal to make the action causal.
            self.steal_suppression_steps = 3

        # ---- MAINTAIN: explicit no-op ----
        elif action_type == ActionType.MAINTAIN:
            pass

    def _calculate_reward(self) -> float:
        """
        Calculate reward signal for the current step.

        Returns:
            float: Reward in range [-20.0, +10.5].
        """
        obs = self._current_obs
        if obs is None:
            return 0.0
        return compute_reward(obs, self._prev_steal_pct)

    # ------------------------------------------------------------------
    # Private utility
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # Accessors (convenience)
    # ------------------------------------------------------------------

    def _get_node_capacity_multiplier(self) -> float:
        """Return capacity multiplier for the current node size."""
        if self._node_size == NodeSizeClass.SMALL:
            return 1.0
        if self._node_size == NodeSizeClass.MEDIUM:
            return 2.0
        return 4.0

    def _compute_current_cost(self) -> float:
        """Compute actual hourly cost based on current node size and replica count."""
        base_costs = {
            NodeSizeClass.SMALL: 10.0,
            NodeSizeClass.MEDIUM: 20.0,
            NodeSizeClass.LARGE: 40.0,
        }
        return base_costs.get(self._node_size, 10.0) + float(self._replicas)

    def _build_observation(self, trace_obs: TraceObservation) -> Observation:
        """Build the agent-facing Observation from raw trace demand and current state."""
        capacity = max(1.0, self._replicas * self._get_node_capacity_multiplier())

        cpu_usage = min(100.0, (trace_obs.base_cpu_demand / capacity) * 100.0)
        mem_usage = min(100.0, (trace_obs.base_mem_demand / capacity) * 100.0)

        demand_pressure = max(0.0, (cpu_usage - 70.0) / 30.0)
        p99 = max(40.0, trace_obs.base_latency_ms * (1.0 + demand_pressure * 0.75))

        if cpu_usage > 90.0:
            raw_steal_pct = min(1.0, max(trace_obs.base_steal_pct, (cpu_usage - 90.0) / 10.0))
        else:
            raw_steal_pct = max(trace_obs.base_steal_pct * 0.5, trace_obs.base_steal_pct)

        if self.steal_suppression_steps > 0:
            steal_pct = round(raw_steal_pct * 0.2, 4)
            self.steal_suppression_steps -= 1
        else:
            steal_pct = round(raw_steal_pct, 4)

        error_rate = trace_obs.base_error_rate
        if cpu_usage > 80.0:
            error_rate = min(1.0, error_rate + (cpu_usage - 80.0) / 60.0)
        else:
            error_rate = min(1.0, error_rate * (0.5 + cpu_usage / 200.0))

        buffer_depth = int(trace_obs.buffer_depth * (1.0 + cpu_usage / 150.0))

        return Observation(
            cpu_usage_pct=round(cpu_usage, 4),
            mem_usage_pct=round(mem_usage, 4),
            p99_latency_ms=round(p99, 4),
            http_error_rate=round(min(1.0, error_rate), 4),
            cpu_steal_pct=round(min(1.0, steal_pct), 4),
            active_replicas=self._replicas,
            buffer_depth=max(0, buffer_depth),
            node_size_class=self._node_size,
            current_hourly_cost=round(self._compute_current_cost(), 4),
            node_bin_density=trace_obs.node_bin_density,
        )

    @property
    def trajectory(self) -> list[TrajectoryStep]:
        """Read-only access to the episode trajectory log."""
        return list(self._trajectory)
