"""
Integration tests for end-to-end inference pipeline (Fix #18).

Tests that the full pipeline works:
1. Environment initialization
2. Episode execution
3. Grading
4. Result validation
"""

import os
import pytest
from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from models import Action, ActionType


@pytest.fixture
def traces_available():
    """Check if trace files exist."""
    traces = [
        "traces/trace_v1_coldstart.json",
        "traces/trace_v1_squeeze.json",
        "traces/trace_v1_entropy.json"
    ]
    return all(os.path.exists(t) for t in traces)


class TestEnvironmentIntegration:
    """Test environment integration and full episodes."""
    
    def test_coldstart_full_episode(self, traces_available):
        """Cold start task should produce valid episode."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_coldstart.json")
        obs = env.reset()
        
        assert obs is not None
        assert obs.active_replicas >= 0
        
        # Run episode
        step_count = 0
        done = False
        while not done and step_count < 100:
            obs, reward, done, info = env.step(Action(action_type=ActionType.SCALE_UP_5))
            step_count += 1
        
        # Verify trajectory
        trajectory = env.trajectory
        assert len(trajectory) > 0, "Should have collected trajectory steps"
        assert all(-20.0 <= step.reward <= 10.5 for step in trajectory)
    
    def test_squeeze_full_episode(self, traces_available):
        """Efficient squeeze task should produce valid episode."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_squeeze.json")
        obs = env.reset()
        
        done = False
        while not done:
            obs, reward, done, info = env.step(Action(action_type=ActionType.MAINTAIN))
        
        trajectory = env.trajectory
        assert len(trajectory) > 0
        grader = EfficientSqueezeGrader()
        score = grader.grade(trajectory)
        assert 0.0 <= score <= 1.0
    
    def test_entropy_full_episode(self, traces_available):
        """Entropy storm task should produce valid episode."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_entropy.json")
        obs = env.reset()
        
        done = False
        step_num = 0
        while not done:
            # Mix of actions to test grader
            if step_num % 5 == 0:
                action = ActionType.REBALANCE_NODE
            else:
                action = ActionType.MAINTAIN
            
            obs, reward, done, info = env.step(Action(action_type=action))
            step_num += 1
        
        trajectory = env.trajectory
        assert len(trajectory) > 0
        grader = EntropyStormGrader()
        score = grader.grade(trajectory)
        assert 0.0 <= score <= 1.0


class TestGraderIntegration:
    """Test graders against real episodic data."""
    
    def test_grader_on_coldstart_trajectory(self, traces_available):
        """ColdStartGrader should grade cold start episodes."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_coldstart.json")
        env.reset()
        
        while not env._step >= len(env.steps_data) - 1:
            env.step(Action(action_type=ActionType.SCALE_UP_5))
        
        grader = ColdStartGrader()
        score = grader.grade(env.trajectory)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_grading_consistency(self, traces_available):
        """Same trajectory should always score the same."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env1 = KubeCostEnv("traces/trace_v1_coldstart.json")
        env1.reset()
        for _ in range(5):
            env1.step(Action(action_type=ActionType.MAINTAIN))
        
        trajectory = env1.trajectory
        grader = ColdStartGrader()
        
        score1 = grader.grade(trajectory)
        score2 = grader.grade(trajectory)  # Grade same trajectory again
        assert score1 == score2, "Grader should be deterministic"


class TestTrajectoryValidation:
    """Test trajectory collection and validation."""
    
    def test_trajectory_has_required_fields(self, traces_available):
        """All trajectory steps should have required fields."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_coldstart.json")
        env.reset()
        
        for _ in range(3):
            env.step(Action(action_type=ActionType.MAINTAIN))
        
        trajectory = env.trajectory
        for step in trajectory:
            assert hasattr(step, 'observation')
            assert hasattr(step, 'action')
            assert hasattr(step, 'reward')
            assert hasattr(step, 'done')
            assert hasattr(step, 'info')
            
            # Verify types
            assert isinstance(step.reward, float)
            assert isinstance(step.done, bool)
            assert isinstance(step.info, dict)
    
    def test_trajectory_reward_bounds(self, traces_available):
        """All rewards in trajectory should be in valid bounds."""
        if not traces_available:
            pytest.skip("Trace files not available")
        
        env = KubeCostEnv("traces/trace_v1_coldstart.json")
        env.reset()
        
        for _ in range(10):
            env.step(Action(action_type=ActionType.SCALE_UP_5))
        
        trajectory = env.trajectory
        for step in trajectory:
            assert -20.0 <= step.reward <= 10.5, f"Reward {step.reward} out of bounds"
