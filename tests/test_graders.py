"""
Comprehensive grader tests (Fix #17).

Tests that graders:
1. Return values in [0.0, 1.0]
2. Handle edge cases (empty trajectories, all violations)
3. Are properly normalized (length-invariant)
4. Prevent passive agents from scoring perfectly
"""

import pytest
from graders import (
    ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader,
    is_healthy_uptime, is_warning_zone, steal_violation,
)
from models import TrajectoryStep, Observation, ActionType, NodeSizeClass


@pytest.fixture
def dummy_observation():
    """Create a dummy healthy observation."""
    return Observation(
        cpu_usage_pct=45.0,
        mem_usage_pct=50.0,
        p99_latency_ms=180.0,
        http_error_rate=0.0,
        cpu_steal_pct=0.05,
        active_replicas=3,
        buffer_depth=5,
        node_size_class=NodeSizeClass.MEDIUM,
        current_hourly_cost=40.0,
        node_bin_density=[0.4] * 10
    )


@pytest.fixture
def dummy_step(dummy_observation):
    """Create a dummy trajectory step."""
    return TrajectoryStep(
        observation=dummy_observation,
        action=ActionType.MAINTAIN,
        reward=1.0,
        done=False,
        info={}
    )


class TestObservationMetrics:
    """Test the observation metrics functions."""
    
    def test_is_healthy_uptime(self):
        assert is_healthy_uptime(280.0) == True
        assert is_healthy_uptime(300.0) == False
        assert is_healthy_uptime(350.0) == False
    
    def test_is_warning_zone(self):
        assert is_warning_zone(200.0) == True
        assert is_warning_zone(250.0) == True
        assert is_warning_zone(299.9) == True
        assert is_warning_zone(300.0) == False
        assert is_warning_zone(150.0) == False
    
    def test_steal_violation(self):
        assert steal_violation(0.20) == True
        assert steal_violation(0.25) == True
        assert steal_violation(0.19) == False
        assert steal_violation(0.0) == False


class TestColdStartGrader:
    """Test Cold Start grader (Task 1)."""
    
    def test_empty_trajectory(self):
        """Empty trajectory should return 0.0."""
        grader = ColdStartGrader()
        assert grader.grade([]) == 0.0
    
    def test_perfect_no_errors(self, dummy_step):
        """Zero errors should return 1.0."""
        grader = ColdStartGrader()
        trajectory = [dummy_step] * 5
        assert grader.grade(trajectory) == 1.0
    
    def test_all_errors(self, dummy_observation):
        """100% error rate should return 0.0."""
        grader = ColdStartGrader()
        obs_with_errors = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0,
            p99_latency_ms=180.0, http_error_rate=1.0,  # 100% errors
            cpu_steal_pct=0.05, active_replicas=3,
            buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        step = TrajectoryStep(
            observation=obs_with_errors,
            action=ActionType.MAINTAIN,
            reward=1.0,
            done=False,
            info={}
        )
        trajectory = [step] * 5
        assert grader.grade(trajectory) == 0.0
    
    def test_partial_errors(self, dummy_observation):
        """50% error rate should return 0.5."""
        grader = ColdStartGrader()
        obs_no_err = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0,
            p99_latency_ms=180.0, http_error_rate=0.0,  # No errors
            cpu_steal_pct=0.05, active_replicas=3,
            buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        obs_with_err = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0,
            p99_latency_ms=180.0, http_error_rate=1.0,  # 100% errors
            cpu_steal_pct=0.05, active_replicas=3,
            buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        step_no_err = TrajectoryStep(observation=obs_no_err, action=ActionType.MAINTAIN, reward=1.0, done=False, info={})
        step_with_err = TrajectoryStep(observation=obs_with_err, action=ActionType.MAINTAIN, reward=1.0, done=False, info={})
        trajectory = [step_no_err, step_with_err]
        assert abs(grader.grade(trajectory) - 0.5) < 0.01
    
    def test_length_invariance(self, dummy_step):
        """Same error rate should produce same score regardless of trajectory length."""
        grader = ColdStartGrader()
        # 2 errors in 5 steps
        trajectory_short = [dummy_step] * 3 + [
            TrajectoryStep(
                observation=Observation(
                    cpu_usage_pct=45.0,  mem_usage_pct=50.0, p99_latency_ms=180.0,
                    http_error_rate=0.4, cpu_steal_pct=0.05, active_replicas=3,
                    buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
                    current_hourly_cost=40.0, node_bin_density=[0.4] * 10
                ),
                action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
            )
        ] * 2
        
        # 4 errors in 10 steps (same rate: 40%)
        trajectory_long = [dummy_step] * 6 + [
            TrajectoryStep(
                observation=Observation(
                    cpu_usage_pct=45.0, mem_usage_pct=50.0, p99_latency_ms=180.0,
                    http_error_rate=0.4, cpu_steal_pct=0.05, active_replicas=3,
                    buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
                    current_hourly_cost=40.0, node_bin_density=[0.4] * 10
                ),
                action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
            )
        ] * 4
        
        score_short = grader.grade(trajectory_short)
        score_long = grader.grade(trajectory_long)
        assert abs(score_short - score_long) < 0.01, "Scores should be length-invariant"


class TestEfficientSqueezeGrader:
    """Test Efficient Squeeze grader (Task 2)."""
    
    def test_empty_trajectory(self):
        """Empty trajectory should return 0.0."""
        grader = EfficientSqueezeGrader()
        assert grader.grade([]) == 0.0
    
    def test_no_violations(self, dummy_step):
        """No violations should return 1.0."""
        grader = EfficientSqueezeGrader()
        trajectory = [dummy_step] * 5  # All have steal < 0.20
        assert grader.grade(trajectory) == 1.0
    
    def test_all_violations(self, dummy_observation):
        """All violations should return 0.0."""
        grader = EfficientSqueezeGrader()
        obs_violation = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0, p99_latency_ms=180.0,
            http_error_rate=0.0, cpu_steal_pct=0.25,  # Violation: >= 0.20
            active_replicas=3, buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        step = TrajectoryStep(
            observation=obs_violation,
            action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
        )
        trajectory = [step] * 5
        assert grader.grade(trajectory) == 0.0
    
    def test_partial_violations(self, dummy_step):
        """Half violations should return ~0.5."""
        grader = EfficientSqueezeGrader()
        obs_violation = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0, p99_latency_ms=180.0,
            http_error_rate=0.0, cpu_steal_pct=0.25,  # Violation
            active_replicas=3, buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        step_violation = TrajectoryStep(
            observation=obs_violation,
            action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
        )
        trajectory = [dummy_step, step_violation, dummy_step, step_violation]
        score = grader.grade(trajectory)
        assert abs(score - 0.5) < 0.01


class TestEntropyStormGrader:
    """Test Entropy Storm grader (Task 3)."""
    
    def test_empty_trajectory(self):
        """Empty trajectory should return 0.0."""
        grader = EntropyStormGrader()
        assert grader.grade([]) == 0.0
    
    def test_no_violations_with_rebalance(self, dummy_step):
        """No violations but with REBALANCE actions should return 1.0 (proactive)."""
        grader = EntropyStormGrader()
        rebalance_step = TrajectoryStep(
            observation=dummy_step.observation,
            action=ActionType.REBALANCE_NODE,  # Proactive action
            reward=1.0, done=False, info={}
        )
        trajectory = [rebalance_step, dummy_step, dummy_step]
        assert grader.grade(trajectory) == 1.0
    
    def test_no_violations_no_rebalance(self, dummy_step):
        """No violations and no REBALANCE should return 0.5 (passive)."""
        grader = EntropyStormGrader()
        trajectory = [dummy_step] * 3  # All maintain, no violations
        assert grader.grade(trajectory) == 0.5
    
    def test_violation_with_proactive_rebalance(self, dummy_step):
        """Violation preceded by REBALANCE should count as prevented."""
        grader = EntropyStormGrader()
        obs_violation = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0, p99_latency_ms=180.0,
            http_error_rate=0.0, cpu_steal_pct=0.25,  # Violation
            active_replicas=3, buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        violation_step = TrajectoryStep(
            observation=obs_violation,
            action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
        )
        rebalance_step = TrajectoryStep(
            observation=dummy_step.observation,
            action=ActionType.REBALANCE_NODE, reward=1.0, done=False, info={}
        )
        
        # REBALANCE 1 step before violation
        trajectory = [dummy_step, rebalance_step, violation_step]
        score = grader.grade(trajectory)
        assert score == 1.0, "Should detect proactive REBALANCE within lookback window"
    
    def test_violation_without_proactive_rebalance(self, dummy_step):
        """Violation without preceding REBALANCE should return 0.0."""
        grader = EntropyStormGrader()
        obs_violation = Observation(
            cpu_usage_pct=45.0, mem_usage_pct=50.0, p99_latency_ms=180.0,
            http_error_rate=0.0, cpu_steal_pct=0.25,  # Violation
            active_replicas=3, buffer_depth=5, node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0, node_bin_density=[0.4] * 10
        )
        violation_step = TrajectoryStep(
            observation=obs_violation,
            action=ActionType.MAINTAIN, reward=1.0, done=False, info={}
        )
        
        trajectory = [dummy_step, dummy_step, violation_step]
        score = grader.grade(trajectory)
        assert score == 0.0, "Should fail to detect proactive action"


class TestGraderBounds:
    """Test all graders produce scores in [0.0, 1.0]."""
    
    def test_all_graders_clamp_output(self, dummy_step):
        """All graders should clamp output to [0.0, 1.0]."""
        graders = [
            ColdStartGrader(),
            EfficientSqueezeGrader(),
            EntropyStormGrader()
        ]
        
        trajectory = [dummy_step] * 10
        for grader in graders:
            score = grader.grade(trajectory)
            assert 0.0 <= score <= 1.0, f"{grader.__class__.__name__} returned {score} out of bounds"
