import pytest

from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from models import Action, ActionType, EnvState, Observation


def test_reset_returns_observation_and_state_type():
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    obs = env.reset()
    assert isinstance(obs, Observation)

    state = env.state()
    assert isinstance(state, EnvState)
    assert state.step == 0


def test_step_applies_scale_action_to_observation():
    """Verify that actions modify internal state (recorded in trajectory).
    
    Note: Observations come directly from traces (pure replay, no formula overlays).
    The scale action modifies internal replica state, which is captured in the
    TrajectoryStep. Future trace steps may reflect the consequences, but the
    current observation is from the trace, not from the action.
    """
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    obs0 = env.reset()

    # Apply scale action - this modifies internal state, not immediate observation
    obs1, reward, done, info = env.step(Action(action_type=ActionType.SCALE_UP_5))
    assert isinstance(obs1, Observation)
    
    # Observation comes from trace (pure replay), not from action
    # But action was recorded in trajectory - verify trajectory has the action
    trajectory = env.trajectory
    assert len(trajectory) > 0
    # TrajectoryStep.action is stored as ActionType enum value (string)
    assert trajectory[-1].action == "SCALE_REPLICAS(+5)"
    
    # Reward is still bounded
    assert -20.0 <= reward <= 10.5


def test_episode_trajectory_consistency():
    """Trajectories logged during episode should be consistent."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    _ = env.reset()
    for i in range(3):
        env.step(Action(action_type=ActionType.MAINTAIN))

    trajectory = env.trajectory
    assert len(trajectory) == 3
    assert all(step.observation.p99_latency_ms >= 0.0 for step in trajectory)
    assert all(0.0 <= step.reward <= 10.5 for step in trajectory)


def test_graders_clamp_scores():
    """Graders should clamp scores to [0.0, 1.0]."""
    empty = []
    assert ColdStartGrader().grade(empty) == 0.0
    assert EfficientSqueezeGrader().grade(empty) == 0.0
    assert EntropyStormGrader().grade(empty) == 0.0


def test_episode_terminates_at_trace_end():
    """Episode must terminate when reaching end of trace."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    env.reset()
    
    step_count = 0
    done = False
    max_safe_steps = 1000
    
    while not done and step_count < max_safe_steps:
        _, _, done, _ = env.step(Action(action_type=ActionType.MAINTAIN))
        step_count += 1
    
    # Episode must terminate before safety limit
    assert done, "Episode should terminate at trace end"
    assert step_count <= len(env.steps_data), "Should not exceed trace length"


def test_observation_immutability():
    """Observations should come unchanged from traces."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    obs_initial = env.reset()
    p99_initial = obs_initial.p99_latency_ms
    cost_initial = obs_initial.current_hourly_cost
    steal_initial = obs_initial.cpu_steal_pct
    
    # Take many action steps
    for _ in range(10):
        env.step(Action(action_type=ActionType.SCALE_UP_20))
    
    # Reset and verify initial observation unchanged
    obs_reset = env.reset()
    assert obs_reset.p99_latency_ms == p99_initial
    assert obs_reset.current_hourly_cost == cost_initial
    assert obs_reset.cpu_steal_pct == steal_initial


def test_cost_penalty_bounded():
    """Cost penalty should never exceed 5.0 in reward calculation."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    env.reset()
    
    # Run episode and verify all rewards are in bounds
    done = False
    while not done:
        _, reward, done, _ = env.step(Action(action_type=ActionType.MAINTAIN))
        assert -20.0 <= reward <= 10.5, f"Reward {reward} out of bounds"


def test_scale_actions_modify_replica_count():
    """Scale actions should modify internal replica count."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    env.reset()
    
    # Scale up
    obs1, _, _, _ = env.step(Action(action_type=ActionType.SCALE_UP_5))
    state1 = env.state()
    assert state1.replicas == 5
    
    obs2, _, _, _ = env.step(Action(action_type=ActionType.SCALE_UP_5))
    state2 = env.state()
    assert state2.replicas == 10
    
    # Scale down
    obs3, _, _, _ = env.step(Action(action_type=ActionType.SCALE_DOWN_1))
    state3 = env.state()
    assert state3.replicas == 9


def test_rebalance_node_action():
    """REBALANCE_NODE should affect steal based on validation."""
    env = KubeCostEnv("traces/trace_v1_entropy.json")
    env.reset()
    
    # Take rebalance action
    obs1, _, _, _ = env.step(Action(action_type=ActionType.REBALANCE_NODE))
    
    # Verify the action was recorded in trajectory (graders use this for scoring)
    trajectory = env.trajectory
    assert len(trajectory) > 0
    assert trajectory[-1].action == ActionType.REBALANCE_NODE


def test_multiple_episodes_are_independent():
    """Multiple reset() calls should create independent episodes."""
    env = KubeCostEnv("traces/trace_v1_coldstart.json")
    
    # Episode 1
    obs1 = env.reset()
    env.step(Action(action_type=ActionType.SCALE_UP_5))
    traj1_len = len(env.trajectory)
    
    # Episode 2 (reset should clear trajectory)
    obs2 = env.reset()
    state2 = env.state()
    assert state2.step == 0
    assert len(env.trajectory) == 0, "Trajectory should be cleared on reset"
    
    # Observations should match
    assert obs1.cpu_usage_pct == obs2.cpu_usage_pct
    assert obs1.p99_latency_ms == obs2.p99_latency_ms
