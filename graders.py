# graders.py
"""
Grader implementations (Phase 4: Grader Spec).

Phase 4 Implementation: Final scoring logic. All grade() stubs replaced with live logic.

Rules enforced across every grader:
  1. Empty trajectory  → return 0.0 explicitly (never divide by zero).
  2. Normalized output → score invariant to trajectory length.
  3. Hard bounds       → final return is always max(0.0, min(1.0, score)).
  4. Float tolerance   → thresholds use >= / < never ==.

Reference: PROJECT_SPEC.md §3 Phase 4 Grader Spec, §5 Audit Fixes, §6 The Three Tasks
"""

import logging
from typing import List

from models import TrajectoryStep, ActionType

__all__ = [
    "ColdStartGrader",
    "EfficientSqueezeGrader",
    "EntropyStormGrader",
    "is_healthy_uptime",
    "is_warning_zone",
    "uptime_score",
    "steal_violation",
    "cost_ratio",
]

# Configure module logger
logger = logging.getLogger(__name__)


# ===== GRADER CONFIGURATION =====


class _GraderConfig:
    """Configuration constants for graders."""

    # SLA thresholds
    SLA_THRESHOLD_MS: float = 300.0
    SLA_WARNING_MIN_MS: float = 200.0

    # Steal thresholds
    STEAL_THRESHOLD: float = 0.20

    # Entropy storm parameters
    LOOKBACK_WINDOW: int = 5
    
    # Cost budget for normalization
    BUDGET: float = 100.0


_CONFIG = _GraderConfig()



# ---------------------------------------------------------------------------
# Observation metrics (module-level functions)
# ---------------------------------------------------------------------------


def is_healthy_uptime(p99_ms: float) -> bool:
    """Check if p99 latency is within SLA."""
    return p99_ms < _CONFIG.SLA_THRESHOLD_MS


def is_warning_zone(p99_ms: float) -> bool:
    """Check if p99 in warning zone [200, 300)."""
    return _CONFIG.SLA_WARNING_MIN_MS <= p99_ms < _CONFIG.SLA_THRESHOLD_MS


def uptime_score(p99_ms: float) -> float:
    """Return 1.0 if healthy, 0.0 if breach."""
    return 1.0 if p99_ms < _CONFIG.SLA_THRESHOLD_MS else 0.0


def steal_violation(steal_pct: float, threshold: float | None = None) -> bool:
    """Check if steal exceeds threshold."""
    if threshold is None:
        threshold = _CONFIG.STEAL_THRESHOLD
    return steal_pct >= threshold


def cost_ratio(hourly_cost: float, budget: float | None = None) -> float:
    """Compute cost as fraction of budget."""
    if budget is None:
        budget = _CONFIG.BUDGET
    return hourly_cost / budget


class ColdStartGrader:
    """
    Task 1: Cold Start (Easy).

    Objective: Scale cluster from 0→5 replicas without SLA breach.

    Formula:
        avg_error_rate = mean(http_error_rate for each step)
        score          = 1.0 - avg_error_rate
        final_score    = max(0.0, min(1.0, score))

    Normalization:
        Uses the *average* error rate — length-invariant by construction.
        A 5-step trace and a 500-step trace with the same mean error rate
        produce the same score.

    Edge case:
        Empty trajectory → return 0.0 explicitly.

    Reference:
        PROJECT_SPEC.md §6 Task 1 Cold Start
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 01: tolerance comparison (< 0.001, not == 0.0)
        Audit Fix 02: normalized by average, not unbounded -= 0.05
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade cold start performance.

        Args:
            trajectory: List of TrajectoryStep instances from the episode.

        Returns:
            float: Score in [0.0, 1.0].
                   1.0 = zero errors throughout.
                   0.0 = 100% error rate throughout (or empty).
        """
        # Edge case: empty trajectory
        if not trajectory:
            logger.warning("ColdStartGrader: received empty trajectory")
            return 0.0

        # Collect http_error_rate from every step's observation
        total_error = sum(
            step.observation.http_error_rate
            for step in trajectory
        )

        # Average across the full episode (length-invariant)
        avg_error_rate = total_error / len(trajectory)

        # Score: perfect uptime = 1.0, total failure = 0.0
        score = 1.0 - avg_error_rate

        # Hard clamp — mathematically cannot exceed [0.0, 1.0]
        return max(0.0, min(1.0, score))


class EfficientSqueezeGrader:
    """
    Task 2: Efficient Squeeze (Medium).

    Objective: Keep cpu_steal_pct < 20% across the full 24-hour load cycle.

    Formula:
        violations  = count(steps where cpu_steal_pct >= 0.20)
        score       = 1.0 - (violations / len(trajectory))
        final_score = max(0.0, min(1.0, score))

    Normalization:
        Divides by trajectory length → violation *rate* is length-invariant.
        10 violations in 100 steps ≡ 20 violations in 200 steps (same score).

    Edge case:
        Empty trajectory → return 0.0 explicitly.

    Float comparison (Audit Fix 01):
        ✗  cpu_steal_pct == 0.20   (float equality — broken)
        ✓  cpu_steal_pct >= 0.20   (threshold comparison — correct)

    Reference:
        PROJECT_SPEC.md §6 Task 2 Efficient Squeeze
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 02: normalized by len(trajectory)
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade efficient squeeze performance.

        Args:
            trajectory: List of TrajectoryStep instances from the episode.

        Returns:
            float: Score in [0.0, 1.0].
                   1.0 = zero steal violations.
                   0.0 = every step was a steal violation.
        """
        # Edge case: empty trajectory
        if not trajectory:
            logger.warning("EfficientSqueezeGrader: received empty trajectory")
            return 0.0

        # Count steps where steal crossed the threshold
        violations = sum(
            1
            for step in trajectory
            if step.observation.cpu_steal_pct >= _CONFIG.STEAL_THRESHOLD
        )

        # Normalized violation rate (length-invariant)
        score = 1.0 - (violations / len(trajectory))

        # Hard clamp
        return max(0.0, min(1.0, score))


class EntropyStormGrader:
    """
    Task 3: Entropy Storm (Hard — Proactive Reasoning).

    Objective: Issue REBALANCE_NODE *before* steal exceeds 20%, using only
               leading indicators. Reactive agents cannot score here.

    Algorithm:
        1. Identify every violation step  (cpu_steal_pct >= 0.20).
        2. If no violations ever occur    → return 0.0  (passive agent).
        3. For each violation at index i:
               Look back through steps [max(0, i − LOOKBACK_WINDOW), i − 1].
               If any step in that window took action REBALANCE_NODE:
                   proactive_actions += 1
        4. Score: `proactive_actions / total_violations`
           final_score = max(0.0, min(1.0, success_rate))

    Lookback window:
        Default = 5 steps. The agent must act within 5 steps before the
        steal spike is observed for the action to count as "proactive".

    Normalization:
        Divides by total_violations — invariant to episode length.
        An agent that predicts 3 out of 3 spikes scores 1.0 whether
        the trace is 10 steps or 1000 steps long.

    Special cases:
        Empty trajectory         → 0.0
        Zero violations          → 0.0  (inaction on hard task is not rewarded)

    Reference:
        PROJECT_SPEC.md §6 Task 3 Entropy Storm
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 04: SCALE_REPLICAS(+20) ensures solvability of hard task
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade entropy storm proactive-rebalancing performance.

        Args:
            trajectory: List of TrajectoryStep instances from the episode.

        Returns:
            float: Score in [0.0, 1.0].
                   1.0 = all violations predicted (or actively prevented via REBALANCE).
                   0.0 = no proactive actions before any violation.
        """
        # Edge case: empty trajectory
        if not trajectory:
            logger.warning("EntropyStormGrader: received empty trajectory")
            return 0.0

        # Step 1: Find all violation indices
        violation_indices = [
            i
            for i, step in enumerate(trajectory)
            if step.observation.cpu_steal_pct >= _CONFIG.STEAL_THRESHOLD
        ]

        # Special case: zero violations
        if not violation_indices:
            # Agent was passive/lucky — only grant partial credit if they took REBALANCE actions
            rebalance_count = sum(1 for step in trajectory if step.action == ActionType.REBALANCE_NODE)
            if rebalance_count > 0:
                # Agent was actively trying to prevent (proactive even when not needed)
                return 1.0
            else:
                # Agent did nothing; don't reward inaction on the hardest task
                return 0.0

        total_violations = len(violation_indices)
        proactive_actions = 0

        # Step 2: For each violation, check the lookback window
        for violation_idx in violation_indices:
            window_start = max(0, violation_idx - _CONFIG.LOOKBACK_WINDOW)
            window_end = violation_idx  # exclusive — we look at steps *before* the breach

            # Did the agent issue REBALANCE_NODE anywhere in the lookback window?
            rebalanced_proactively = any(
                trajectory[j].action == ActionType.REBALANCE_NODE
                for j in range(window_start, window_end)
            )

            if rebalanced_proactively:
                proactive_actions += 1

        # Step 3: Normalized success rate
        success_rate = proactive_actions / total_violations

        # Hard clamp
        return max(0.0, min(1.0, success_rate))
