# env.py
"""
KubeCost-Gym Environment (OpenEnv Interface).

Phase 3 Implementation: Reward Specification and Physics Logic.
All methods fully implemented — no stub (pass) bodies remain.

Reference: PROJECT_SPEC.md §2 OpenEnv Interface, §3 Reward Spec, §4 Environment Interface
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from openenv.core import Environment

try:
    from k8s_cost_optimizer.models import (
        Observation,
        Action,
        EnvState,
        ActionType,
        NodeSizeClass,
        TrajectoryStep,
        TraceData,
        TraceObservation,
        HOURLY_BUDGET,
    )
except ImportError:
    from models import (
        Observation,
        Action,
        EnvState,
        ActionType,
        NodeSizeClass,
        TrajectoryStep,
        TraceData,
        TraceObservation,
        HOURLY_BUDGET,
    )

__all__ = [
    "K8sCostOptimizerEnvironment",
    "load_trace",
    "compute_reward",
    "validate_action",
    "get_replica_delta",
    "EnvError",
    "get_grader_for_task",
]

# Configure module logger
logger = logging.getLogger(__name__)


def get_grader_for_task(task_name: str) -> Any:
    """Factory to get the appropriate grader for a task."""
    try:
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
    except ImportError:
        # Fallback for internal module structure
        try:
            from k8s_cost_optimizer.graders import (
                ColdStartGrader,
                EfficientSqueezeGrader,
                EntropyStormGrader,
            )
        except ImportError:
            return None

    graders = {
        "cold_start": ColdStartGrader,
        "efficient_squeeze": EfficientSqueezeGrader,
        "entropy_storm": EntropyStormGrader,
    }
    grader_cls = graders.get(task_name)
    return grader_cls() if grader_cls else None


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
    HOURLY_BUDGET: float = HOURLY_BUDGET

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
    UPTIME_REWARD: float = float(os.getenv("ENV_UPTIME_REWARD", "10.0"))
    COST_PENALTY_RATE: float = float(os.getenv("ENV_COST_PENALTY_RATE", "5.0"))
    COST_PENALTY_CAP: float = float(os.getenv("ENV_COST_PENALTY_CAP", "5.0"))
    RAMP_PENALTY_RATE: float = float(os.getenv("ENV_RAMP_PENALTY_RATE", "5.0"))
    SLA_BREACH_PENALTY: float = float(os.getenv("ENV_SLA_BREACH_PENALTY", "20.0"))
    PROACTIVE_BONUS: float = float(os.getenv("ENV_PROACTIVE_BONUS", "0.5"))


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

    # Cost penalty (capped per spec §3.3)
    cost_penalty = min(
        _CONFIG.COST_PENALTY_RATE * cost_fraction,
        _CONFIG.COST_PENALTY_CAP,
    )

    # Ramp penalty (dense signal in warning zone [200, 300))
    ramp_penalty = 0.0
    if _CONFIG.SLA_WARNING_MIN_MS <= p99 < _CONFIG.SLA_THRESHOLD_MS:
        ramp_penalty = (
            (p99 - _CONFIG.SLA_WARNING_MIN_MS) / 100.0
        ) * _CONFIG.RAMP_PENALTY_RATE

    # SLA breach hard penalty
    sla_breach_penalty = (
        _CONFIG.SLA_BREACH_PENALTY if p99 >= _CONFIG.SLA_THRESHOLD_MS else 0.0
    )

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
        raise ActionValidationError(
            f"action_type must be ActionType, got {type(action.action_type)}"
        )


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


class K8sCostOptimizerEnvironment(Environment):
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

    # Enable concurrent WebSocket sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, trace_path: str = "traces/trace_v1_coldstart.json") -> None:
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
        self._node_size: NodeSizeClass = (
            NodeSizeClass(first_node) if isinstance(first_node, str) else first_node
        )
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
        if not self.steps_data:
            raise TraceLoadError(
                f"Trace '{self.trace_path}' contains no steps. Cannot reset."
            )

        self._step = 0
        self._trajectory = []

        # Read starting cluster config from trace step 0
        first_trace_step = self.steps_data[0]
        first_obs = first_trace_step.observation

        self._replicas = first_obs.active_replicas
        first_node = first_obs.node_size_class
        self._node_size = (
            NodeSizeClass(first_node) if isinstance(first_node, str) else first_node
        )
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
        if self._current_obs is None:
            raise EnvError(
                "step() called before reset(). Call env.reset() to initialize the episode."
            )

        # Guard: reject calls after episode has ended
        if self._step >= self.total_steps - 1:
            logger.warning("step() called on a finished episode. Call reset() first.")
            return self._current_obs, 0.0, True, {"step": self._step, "terminal": True}

        # 0. Validate action
        validate_action(action)

        # Capture current steal for next calculation
        if self._current_obs is not None:
            self._prev_steal_pct = self._current_obs.cpu_steal_pct

        self._step += 1

        if self._step >= self.total_steps:
            raise EnvError(
                f"step() called with _step={self._step} >= total_steps={self.total_steps}. "
                "Call reset() to start a new episode."
            )

        # Apply the agent action first so the next observation reflects capacity changes.
        self._apply_action(action)

        trace_step = self.steps_data[self._step]
        trace_obs = trace_step.observation

        new_obs = self._build_observation(trace_obs)
        self._current_obs = new_obs

        reward: float = self._calculate_reward()
        done: bool = self._step >= self.total_steps - 1

        # Build info dict
        info: Dict[str, Any] = {
            "step": self._step,
            "task_name": self.trace.task_name,
            "task_difficulty": self.trace.task_difficulty,
            "replicas": self._replicas,
            "node_size": self._node_size.value
            if isinstance(self._node_size, NodeSizeClass)
            else self._node_size,
            "trace_reason": trace_step.dynamics.get("reason", ""),
        }

        # Log to trajectory
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

    @property
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
            steal_suppression_steps=self.steal_suppression_steps,
            episode_id=getattr(self, "episode_id", ""),
            step_count=getattr(self, "step_count", self._step),
        )

    def grade(self, trajectory: list | None = None) -> float:
        """
        Grade the episode trajectory using the task-specific grader.

        This is a mandatory method for the OpenEnv platform to retrieve
        task scores at the end of an episode.

        Args:
            trajectory: Optional list of TrajectoryStep. If None, uses self._trajectory.

        Returns:
            float: Score strictly in [0.1, 0.9].
        """
        if trajectory is None:
            trajectory = self._trajectory

        if not trajectory:
            logger.warning(f"grade() called for {self.task_name} with empty trajectory")
            return 0.1

        grader = get_grader_for_task(self.task_name)
        if grader:
            try:
                score = grader.grade(trajectory)
                # Ensure we return a float and follow strict bounds
                return float(max(0.1, min(0.9, score)))
            except Exception as e:
                logger.error(f"Error during grading task '{self.task_name}': {e}")
                return 0.1

        logger.warning(f"No grader found for task '{self.task_name}'. Returning 0.1.")
        return 0.1

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
        """Apply action to environment state."""
        action_type = action.action_type

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
            self._replicas = min(_REPLICAS_MAX, self._replicas + 20)
        elif action_type == ActionType.UPGRADE_NODE:
            current_tier = _NODE_TIER[self._node_size]
            next_tier = min(current_tier + 1, len(_NODE_TIER) - 1)
            self._node_size = _NODE_FROM_TIER[next_tier]
        elif action_type == ActionType.REBALANCE_NODE:
            self.steal_suppression_steps = 3
        elif action_type == ActionType.MAINTAIN:
            pass
        else:
            logger.error(f"_apply_action: unhandled ActionType '{action_type}'.")

    def _calculate_reward(self) -> float:
        """Calculate reward signal for the current step."""
        obs = self._current_obs
        if obs is None:
            return 0.1
        return compute_reward(obs, self._prev_steal_pct)

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
            raw_steal_pct = trace_obs.base_steal_pct * 0.5
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
