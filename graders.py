# graders.py
"""
Grader implementations (Phase 4: Grader Spec).

Each grader:
  1. Has mathematical formula (documented in docstring)
  2. Returns float in [0.0, 1.0] (CRITICAL: must be bounded)
  3. Normalizes by trajectory length (not unbounded accumulation)
  4. Handles empty trajectory explicitly

Reference: PROJECT_SPEC.md §3 Phase 4, §6 The Three Tasks
"""

from typing import List
from models import TrajectoryStep


class ColdStartGrader:
    """
    Task 1: Cold Start (Easy).
    
    Objective: Scale cluster from 0→5 replicas without SLA breach.
    
    Formula (mathematical notation):
        score = 1.0 - http_error_rate_avg
        final_score = max(0.0, min(1.0, score))
    
    Normalization:
        By average error rate (length-invariant)
        Same error rate → same score regardless of episode length
    
    Edge case:
        Empty trajectory → return 0.0 explicitly
    
    Reference:
        PROJECT_SPEC.md §6 Task 1 Cold Start
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 02: Normalized (not unbounded -= 0.05)
    
    Common Failure (spec audit fix 02):
        ✗ score = 1.0; score -= 0.05 per violation
           (unbounded, length-dependent)
        ✓ score = 1.0 - (violations / len(trajectory))
           (normalized, length-invariant)
    """
    
    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade cold start performance.
        
        Args:
            trajectory: List of trajectory steps
        
        Returns:
            float: Score in [0.0, 1.0]
        
        Implementation plan:
            1. Check: if not trajectory: return 0.0
            2. Iterate through trajectory steps
            3. Collect http_error_rate from each observation
            4. Compute average error rate
            5. score = 1.0 - avg_error_rate
            6. Return max(0.0, min(1.0, score))
        """
        # STUB: Compute and return normalized score
        pass


class EfficientSqueezeGrader:
    """
    Task 2: Efficient Squeeze (Medium).
    
    Objective: Maintain cpu_steal_pct < 20% across 24-hour sine-wave load cycle.
    
    Formula (mathematical notation):
        violations = count(steps where cpu_steal_pct >= 0.20)
        score = 1.0 - (violations / len(trajectory))
        final_score = max(0.0, min(1.0, score))
    
    Normalization:
        By violation count per trajectory length
        Same violation rate → same normalized score
        E.g., 10 violations in 100 steps = 20 violations in 200 steps
    
    Edge case:
        Empty trajectory → return 0.0 explicitly
    
    Reference:
        PROJECT_SPEC.md §6 Task 2 Efficient Squeeze
        PROJECT_SPEC.md §3 Phase 4 Grader Spec
        Audit Fix 02: Normalized by len(trajectory)
    
    Key Insight:
        Violation rate (0.0 = perfect, 1.0 = all steps violated) is invariant
        to trajectory length. This ensures fair comparison across different
        simulation durations (24h vs 48h traces).
    
    Float Comparison (spec audit fix 01):
        ✗ cpu_steal_pct == 0.20
        ✓ cpu_steal_pct >= 0.20 (or < 0.20 for threshold)
    """
    
    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade efficient squeeze performance.
        
        Args:
            trajectory: List of trajectory steps
        
        Returns:
            float: Score in [0.0, 1.0]
        
        Implementation plan:
            1. Check: if not trajectory: return 0.0
            2. Scan trajectory: count steps where cpu_steal_pct >= 0.20
            3. violations = count
            4. score = 1.0 - (violations / len(trajectory))
            5. Return max(0.0, min(1.0, score))
        """
        # STUB: Compute and return normalized score
        pass


class EntropyStormGrader:
    """
    Task 3: Entropy Storm (Hard).
    
    Objective: Issue REBALANCE_NODE BEFORE steal exceeds 20% (proactive reasoning).
    
    Formula (mathematical notation):
        1. Identify violations: steps where cpu_steal_pct >= 0.20
        2. For each violation at step i:
           - Check if REBALANCE_NODE occurred in steps [max(0, i-k), i-1]
             (lookback window of k steps; spec: k=TBD)
           - If yes: proactive_count += 1
           - If no: failure_count += 1
        3. success_rate = proactive_count / max(1, total_violations)
        4. score = success_rate × 1.0 + cost_bonus
        5. final_score = max(0.0, min(1.0, score))
    
    Special cases:
        - No violations (steal never >= 0.20): score = 1.0 (agent won)
        - Empty trajectory: score = 0.0
    
    Normalization:
        By count of total violations
        Same prediction accuracy → same score
    
    Reference:
        PROJECT_SPEC.md §6 Task 3 Entropy Storm
        PROJECT_SPEC.md §3 Phase 4 Grader Spec, §5 Audit Fix 04
    
    Key Insight (Audit Fix 04 & Task Design):
        This is the ONLY task where reactive scaling (AFTER p99 breach)
        cannot achieve high score. Reactive agents see violation AFTER it happens;
        cannot undo it. Agent MUST learn to predict and act BEFORE the
        leading indicator (cpu_steal_pct) rises above 20%.
        
        Tests for genuine proactive reasoning, not just reactive optimization.
    
    Design Challenge:
        If ActionType missing SCALE_REPLICAS(+20), hard task becomes structurally
        unsolvable (no action sequence achieves 5x replica increase needed for
        emergency burst). Verify ActionType complete before implementation.
    """
    
    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade entropy storm (proactive rebalancing) performance.
        
        Args:
            trajectory: List of trajectory steps
        
        Returns:
            float: Score in [0.0, 1.0]
        
        Implementation plan:
            1. Check: if not trajectory: return 0.0
            2. Identify violations: cpu_steal_pct >= 0.20
            3. If no violations: return 1.0 (agent avoided problem)
            4. For each violation:
               - Look back up to k steps for REBALANCE_NODE action
               - Count successful proactive actions
            5. success_rate = proactive_actions / total_violations
            6. score = success_rate (no bonus for reactive scaling in this task)
            7. Return max(0.0, min(1.0, score))
        """
        # STUB: Compute and return normalized score
        pass
