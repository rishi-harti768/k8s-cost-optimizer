# graders.py
"""
Grader implementations (Phase 4: Grader Spec).

Phase 2 Implementation: All grade() stubs replaced with live scoring logic.

Rules enforced across every grader:
  1. Empty trajectory  → return 0.0 explicitly (never divide by zero).
  2. Normalized output → score invariant to trajectory length.
  3. Hard bounds       → final return is always max(0.0, min(1.0, score)).
  4. Float tolerance   → thresholds use >= / < never ==.

Reference: PROJECT_SPEC.md §3 Phase 4 Grader Spec, §5 Audit Fixes, §6 The Three Tasks
"""

from typing import List
from models import TrajectoryStep, ActionType


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

    # Steal threshold defined once — easy to update if spec changes.
    STEAL_THRESHOLD: float = 0.20

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
            return 0.0

        # Count steps where steal crossed the 20% threshold
        violations = sum(
            1
            for step in trajectory
            if step.observation.cpu_steal_pct >= self.STEAL_THRESHOLD
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
        2. If no violations ever occur    → return 1.0  (agent won outright).
        3. For each violation at index i:
               Look back through steps [max(0, i − LOOKBACK_WINDOW), i − 1].
               If any step in that window took action REBALANCE_NODE:
                   proactive_actions += 1
        4. success_rate = proactive_actions / total_violations
        5. final_score  = max(0.0, min(1.0, success_rate))

    Lookback window:
        Default = 5 steps. The agent must act within 5 steps before the
        steal spike is observed for the action to count as "proactive".

    Normalization:
        Divides by total_violations — invariant to episode length.
        An agent that predicts 3 out of 3 spikes scores 1.0 whether
        the trace is 10 steps or 1000 steps long.

    Special cases:
        Empty trajectory         → 0.0
        Zero violations          → 1.0  (agent prevented every breach)

    Reference:
        PROJECT_SPEC.md §6 Task 3 Entropy Storm
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 04: SCALE_REPLICAS(+20) ensures solvability of hard task
    """

    # How many steps before a violation a REBALANCE_NODE counts as proactive.
    LOOKBACK_WINDOW: int = 5

    # Steal threshold for defining a "violation"
    STEAL_THRESHOLD: float = 0.20

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade entropy storm proactive-rebalancing performance.

        Args:
            trajectory: List of TrajectoryStep instances from the episode.

        Returns:
            float: Score in [0.0, 1.0].
                   1.0 = all violations predicted (or no violations at all).
                   0.0 = no proactive actions before any violation.
        """
        # Edge case: empty trajectory
        if not trajectory:
            return 0.0

        # Step 1: Find all violation indices
        violation_indices = [
            i
            for i, step in enumerate(trajectory)
            if step.observation.cpu_steal_pct >= self.STEAL_THRESHOLD
        ]

        # Special case: zero violations → agent avoided every breach → perfect score
        if not violation_indices:
            return 1.0

        total_violations = len(violation_indices)
        proactive_actions = 0

        # Step 2: For each violation, check the lookback window
        for violation_idx in violation_indices:
            window_start = max(0, violation_idx - self.LOOKBACK_WINDOW)
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
