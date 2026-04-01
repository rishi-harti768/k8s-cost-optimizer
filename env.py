# env.py
"""
KubeCost-Gym Environment (OpenEnv Interface).

Phase 1 Implementation: Physics Engine.
All methods fully implemented — no stub (pass) bodies remain.

Reference: PROJECT_SPEC.md §2 OpenEnv Interface, §3 Reward Spec, §4 Environment Interface
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any

from models import (
    Observation, Action, EnvState,
    ActionType, NodeSizeClass, TrajectoryStep,
)

# ---------------------------------------------------------------------------
# Node-size ordering for UPGRADE_NODE logic (irreversible for 1 step).
# ---------------------------------------------------------------------------
_NODE_TIER = {
    NodeSizeClass.SMALL: 0,
    NodeSizeClass.MEDIUM: 1,
    NodeSizeClass.LARGE: 2,
}
_NODE_FROM_TIER = {v: k for k, v in _NODE_TIER.items()}

# Budget used in cost penalty: cost fraction = current_hourly_cost / BUDGET
_HOURLY_BUDGET = 100.0

# Reward bounds (spec §3.3)
_R_MIN = -20.0
_R_MAX = 10.5

# Replica hard bounds
_REPLICAS_MIN = 0
_REPLICAS_MAX = 200


class KubeCostEnv:
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
        self.trace: dict = self._load_trace(self.trace_path)
        self.steps_data = self.trace["steps"]  # list[dict]
        self.total_steps: int = len(self.steps_data)

        # ------------------------------------------------------------------
        # Episode state — populated properly by reset()
        # ------------------------------------------------------------------
        self._step: int = 0
        self._current_obs: Observation | None = None

        # Mutable cluster state (updated by _apply_action)
        first_obs_raw = self.steps_data[0]["observation"]
        self._replicas: int = int(first_obs_raw["active_replicas"])
        self._node_size: NodeSizeClass = NodeSizeClass(
            first_obs_raw["node_size_class"]
        )
        self._prev_steal_pct: float = float(first_obs_raw["cpu_steal_pct"])

        # Trajectory log (list[TrajectoryStep]) — filled during step()
        self._trajectory: list[TrajectoryStep] = []

    # ------------------------------------------------------------------
    # OpenEnv Public Interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
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
        first = self.steps_data[0]
        obs_raw = first["observation"]

        self._replicas = int(obs_raw["active_replicas"])
        self._node_size = NodeSizeClass(obs_raw["node_size_class"])
        self._prev_steal_pct = float(obs_raw["cpu_steal_pct"])

        # Build typed Observation from trace data
        self._current_obs = self._parse_observation(obs_raw)
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
            1. Apply action  → mutate self._replicas / self._node_size.
            2. Advance step counter (next trace row provides new observation).
            3. Build new Observation from trace (physics ground-truth).
            4. Compute reward from new observation.
            5. Determine done.
            6. Return 4-tuple.
        """
        # 1. Apply action — mutates self._replicas / self._node_size
        self._apply_action(action)

        # 2. Advance step counter
        self._step += 1

        # 3. Load next observation from trace (deterministic physics)
        #    If we've gone beyond the trace length, replay last step.
        trace_idx = min(self._step, self.total_steps - 1)
        obs_raw = self.steps_data[trace_idx]["observation"]
        self._current_obs = self._parse_observation(obs_raw)

        # 4. Compute reward (from updated obs)
        reward: float = self._calculate_reward()

        # 5. Determine done
        done: bool = self._step >= self.total_steps - 1

        # 6. Build info dict
        info: Dict[str, Any] = {
            "step": self._step,
            "task_name": self.trace.get("task_name", ""),
            "task_difficulty": self.trace.get("task_difficulty", ""),
            "replicas": self._replicas,
            "node_size": self._node_size.value,
            "trace_reason": self.steps_data[trace_idx].get("dynamics", {}).get("reason", ""),
        }

        # 7. Log to trajectory
        self._trajectory.append(
            TrajectoryStep(
                observation=self._current_obs,
                action=action.action_type,
                reward=reward,
                done=done,
                info=info,
                uptime_metric=1.0 if self._current_obs.p99_latency_ms < 300.0 else 0.0,
                cost_metric=self._current_obs.current_hourly_cost / _HOURLY_BUDGET,
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

        In all cases, saves the pre-action steal_pct for the proactive bonus check
        at reward time, then records the action for trajectory logging.

        Reference: PROJECT_SPEC.md §4 Action Space
        """
        # Snapshot steal_pct from current observation *before* mutation
        if self._current_obs is not None:
            self._prev_steal_pct = self._current_obs.cpu_steal_pct

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

        # ---- REBALANCE_NODE: proactive signal, no structural change ----
        elif action_type == ActionType.REBALANCE_NODE:
            # State unchanged; bonus computed in _calculate_reward()
            pass

        # ---- MAINTAIN: explicit no-op ----
        elif action_type == ActionType.MAINTAIN:
            pass

    def _calculate_reward(self) -> float:
        """
        Calculate reward signal for the current step (internal helper).

        Implements the exact formula from PROJECT_SPEC.md §3 (Phase 3 Reward Spec):

            R = (10.0 × Uptime)
              − (5.0 × Cost/Budget)
              − RampPenalty(p99)
              − SLABreach(p99)
              + ProactiveBonus

        Component definitions:
            Uptime         = 1.0  if p99 < 300ms, else 0.0
            RampPenalty    = (p99 − 200) / 100 × 5.0    when p99 ∈ [200, 300)  ← Audit Fix 03
            SLABreach      = 20.0 penalty                when p99 ≥ 300ms
            ProactiveBonus = +0.5 when steal drops AND p99 < 300ms

        Bounds: R clamped to [R_MIN, R_MAX] = [-20.0, +10.5].

        Returns:
            float: Reward in range [-20.0, +10.5].
        """
        obs = self._current_obs
        if obs is None:
            return 0.0

        p99 = obs.p99_latency_ms
        cost_fraction = obs.current_hourly_cost / _HOURLY_BUDGET

        # ---- Uptime component ----
        uptime = 1.0 if p99 < 300.0 else 0.0
        uptime_reward = 10.0 * uptime

        # ---- Cost penalty ----
        cost_penalty = 5.0 * cost_fraction

        # ---- Ramp penalty (Audit Fix 03: eliminates cliff at 300ms) ----
        #   Linear gradient in [200, 300) — gives dense signal in warning zone.
        ramp_penalty = 0.0
        if 200.0 <= p99 < 300.0:
            ramp_penalty = ((p99 - 200.0) / 100.0) * 5.0  # [0.0, 5.0)

        # ---- SLA breach hard penalty ----
        sla_breach_penalty = 20.0 if p99 >= 300.0 else 0.0

        # ---- Proactive bonus ----
        #   Granted when steal_pct is *dropping* compared to previous step
        #   AND p99 is still healthy (< 300ms).
        proactive_bonus = 0.0
        steal_dropped = obs.cpu_steal_pct < self._prev_steal_pct
        if steal_dropped and p99 < 300.0:
            proactive_bonus = 0.5

        # ---- Sum and clamp ----
        raw_reward = (
            uptime_reward
            - cost_penalty
            - ramp_penalty
            - sla_breach_penalty
            + proactive_bonus
        )
        return float(max(_R_MIN, min(_R_MAX, raw_reward)))

    def _load_trace(self, trace_path: Path) -> dict:
        """
        Load and validate deterministic trace JSON (internal helper).

        Args:
            trace_path: Path to JSON file.

        Returns:
            dict: Validated trace structure.

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

        Raises:
            FileNotFoundError: If trace_path does not exist.
            ValueError: If required schema keys are missing or steps list is empty.
        """
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Trace file not found: {trace_path}. "
                f"Expected one of: traces/trace_v1_coldstart.json, "
                f"traces/trace_v1_squeeze.json, traces/trace_v1_entropy.json"
            )

        with trace_path.open("r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        # ---- Required top-level keys ----
        missing_keys = [k for k in ("task_name", "task_difficulty", "steps") if k not in data]
        if missing_keys:
            raise ValueError(
                f"Trace JSON missing required keys: {missing_keys} "
                f"(file: {trace_path})"
            )

        # ---- steps must be a non-empty list ----
        if not isinstance(data["steps"], list) or len(data["steps"]) == 0:
            raise ValueError(
                f"Trace 'steps' must be a non-empty list (file: {trace_path})"
            )

        # ---- Validate each step has required sub-keys ----
        for i, step in enumerate(data["steps"]):
            for sub_key in ("step", "observation"):
                if sub_key not in step:
                    raise ValueError(
                        f"Trace step index {i} missing key '{sub_key}' "
                        f"(file: {trace_path})"
                    )
            obs = step["observation"]
            required_obs_keys = [
                "cpu_usage_pct", "mem_usage_pct", "p99_latency_ms",
                "http_error_rate", "cpu_steal_pct", "active_replicas",
                "buffer_depth", "node_size_class", "current_hourly_cost",
                "node_bin_density",
            ]
            missing_obs = [k for k in required_obs_keys if k not in obs]
            if missing_obs:
                raise ValueError(
                    f"Trace step {i} observation missing keys: {missing_obs} "
                    f"(file: {trace_path})"
                )
            if not isinstance(obs["node_bin_density"], list) or len(obs["node_bin_density"]) != 10:
                raise ValueError(
                    f"Trace step {i}: node_bin_density must be a 10-element list "
                    f"(file: {trace_path})"
                )

        return data

    # ------------------------------------------------------------------
    # Private utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_observation(obs_raw: dict) -> Observation:
        """Build a typed Observation Pydantic model from a raw trace obs dict."""
        return Observation(
            cpu_usage_pct=float(obs_raw["cpu_usage_pct"]),
            mem_usage_pct=float(obs_raw["mem_usage_pct"]),
            p99_latency_ms=float(obs_raw["p99_latency_ms"]),
            http_error_rate=float(obs_raw["http_error_rate"]),
            cpu_steal_pct=float(obs_raw["cpu_steal_pct"]),
            active_replicas=int(obs_raw["active_replicas"]),
            buffer_depth=int(obs_raw["buffer_depth"]),
            node_size_class=NodeSizeClass(obs_raw["node_size_class"]),
            current_hourly_cost=float(obs_raw["current_hourly_cost"]),
            node_bin_density=[float(v) for v in obs_raw["node_bin_density"]],
        )

    # ------------------------------------------------------------------
    # Accessors (convenience)
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> list[TrajectoryStep]:
        """Read-only access to the episode trajectory log."""
        return list(self._trajectory)
