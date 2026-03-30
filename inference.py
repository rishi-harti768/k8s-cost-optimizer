# inference.py
"""
LLM Inference Pipeline for KubeCost-Gym (Google Gemini 2.5 Flash).

Location: ROOT directory (spec §5 requirement).

Environment variables (required):
  - GOOGLE_API_KEY: Google Generative AI API key (Gemini)
  - MODEL_NAME: Model identifier (default: "gemini-2.5-flash")
  - HF_TOKEN: HuggingFace API token for Space submission

Runtime requirement: Complete end-to-end in <20 minutes.

Reference: PROJECT_SPEC.md §5 Infra Spec, §7 Inference Contract
            (Modified to use Google Gemini instead of OpenAI per spec override)
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from models import Observation, Action, ActionType, TrajectoryStep, Trajectory


# ===== ENVIRONMENT VARIABLE VALIDATION =====

def get_env_or_raise(key: str, default: str = None) -> str:
    """
    Get environment variable or raise if missing and no default.
    
    Args:
        key: Environment variable name
        default: Default value if not found (optional)
    
    Returns:
        str: Environment variable value
    
    Raises:
        ValueError: If key not found and no default provided
    """
    value = os.getenv(key, default)
    if not value:
        raise ValueError(f"Missing required env var: {key}")
    return value


def validate_env():
    """Validate all required environment variables are set."""
    required = ["GOOGLE_API_KEY", "HF_TOKEN"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")


# ===== INFERENCE PIPELINE =====

class CostOptimizerAgent:
    """
    LLM-based decision agent for cost optimization (Google Gemini backend).
    
    Responsibilities:
        - Observe environment state (Observation model)
        - Query Gemini LLM for action recommendation (JSON response)
        - Validate response, extract ActionType
        - Execute step, collect trajectory
        - Score trajectory with graders
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        """
        Initialize Gemini LLM inference client.
        
        Args:
            model_name: Model ID (default: "gemini-2.5-flash")
            api_key: Google API key (from GOOGLE_API_KEY env var)
        
        Implementation plan:
            - Import google.generativeai
            - Configure API key
            - Initialize generative model with model_name
            - Validate model availability
            - Store for inference calls
        """
        self.model_name = model_name or "gemini-2.5-flash"
        self.api_key = api_key or get_env_or_raise("GOOGLE_API_KEY")
        # STUB: Initialize Google Generative AI client
        pass
    
    def decide(self, observation: Observation, task_description: str = "") -> Action:
        """
        Query Gemini LLM for action given current observation.
        
        Args:
            observation: Current Observation model
            task_description: Task context for LLM reasoning
        
        Returns:
            Action: Selected action with validation
        
        Implementation plan:
            - Serialize observation to JSON for context
            - Construct prompt with:
              * Current observation state
              * Task description and objective
              * Available actions (ActionType enum values)
              * Reward signal guidance (minimize cost, maintain SLA, etc.)
            - Query Gemini with response_format="json" (if supported)
            - Parse JSON response for action_type field
            - Validate response is valid ActionType member
            - Return Action(action_type=...) on success
            - Return Action(action_type=ActionType.MAINTAIN) on parse failure
        
        Error handling:
            - If Gemini response invalid: default to MAINTAIN
            - Log failures for debugging
        """
        # STUB: Call Gemini, parse response, validate, return Action
        pass
    
    def run_episode(self, env: KubeCostEnv, max_steps: int = 1000, task_name: str = "") -> Trajectory:
        """
        Run one full episode with Gemini agent.
        
        Args:
            env: Environment instance (KubeCostEnv)
            max_steps: Max steps before forced termination
            task_name: Task identifier for Gemini context
        
        Returns:
            Trajectory: List of trajectory steps for grading
        
        Implementation plan:
            1. Call env.reset() to get initial_obs
            2. Initialize trajectory = []
            3. Loop until done or max_steps:
               - obs_current = current observation
               - Call decide(obs_current, task_name) → action
               - Call env.step(action) → (obs_next, reward, done, info)
               - Create TrajectoryStep:
                 * observation=obs_current
                 * action=action.action_type
                 * reward=reward
                 * done=done
                 * info=info
                 * uptime_metric=calculate from obs
                 * cost_metric=calculate from obs
               - Append to trajectory
               - If done: break
            4. Return Trajectory(steps=trajectory)
        """
        # STUB: Run full episode, collect trajectory
        pass
    
    def evaluate_task(self, env: KubeCostEnv, task_name: str, grader) -> float:
        """
        Run episode and score with appropriate grader.
        
        Args:
            env: Environment instance
            task_name: Task identifier ("cold_start", "efficient_squeeze", "entropy_storm")
            grader: Grader instance (ColdStartGrader, etc.)
        
        Returns:
            float: Score from grader [0.0, 1.0]
        
        Implementation plan:
            - Run episode with run_episode()
            - Call grader.grade(trajectory.steps)
            - Return score
        """
        # STUB: Run episode and grade
        pass


# ===== MAIN INFERENCE ENTRY POINT =====

def main():
    """
    Main inference entry point.
    
    Spec requirements (PROJECT_SPEC.md §5 Infra Spec):
        - Runs when executed: python inference.py
        - Must complete in <20 minutes
        - Reads GOOGLE_API_KEY, HF_TOKEN from environment
        - Uses Google Gemini LLM
        - Outputs results (trajectories, scores)
    
    Workflow:
        1. Validate environment variables
        2. Initialize agent and environments
        3. Run 3 tasks:
           - cold_start.json with ColdStartGrader
           - efficient_squeeze.json with EfficientSqueezeGrader
           - entropy_storm.json with EntropyStormGrader
        4. Collect trajectories and scores
        5. Output results to console/file
        6. Exit successfully
    
    Implementation plan:
        - Call validate_env()
        - Initialize CostOptimizerAgent
        - For each task:
          * Create KubeCostEnv(traces/trace_v1_{task_name}.json)
          * Select appropriate grader
          * Run env.reset()
          * Execute agent loop
          * Score episode
          * Log results
        - Print summary (scores per task)
        - Exit code 0 on success, 1 on failure
    """
    
    try:
        # Validate environment
        validate_env()
        print("✓ Environment variables validated")
        
        # Initialize agent
        api_key = get_env_or_raise("GOOGLE_API_KEY")
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        agent = CostOptimizerAgent(model_name=model_name, api_key=api_key)
        print(f"✓ Agent initialized with model: {model_name}")
        
        # Define tasks
        tasks = [
            {
                "name": "cold_start",
                "trace": "traces/trace_v1_coldstart.json",
                "grader": ColdStartGrader(),
                "description": "Scale cluster from 0→5 replicas without SLA breach"
            },
            {
                "name": "efficient_squeeze",
                "trace": "traces/trace_v1_squeeze.json",
                "grader": EfficientSqueezeGrader(),
                "description": "Maintain <20% steal over 24-hour load cycle"
            },
            {
                "name": "entropy_storm",
                "trace": "traces/trace_v1_entropy.json",
                "grader": EntropyStormGrader(),
                "description": "Proactive REBALANCE_NODE before steal>20%"
            }
        ]
        
        results = {}
        
        # Run inference on each task
        for task in tasks:
            print(f"\n[{task['name'].upper()}]")
            print(f"  Description: {task['description']}")
            
            # STUB: Initialize environment, run episode, score
            #
            # try:
            #     env = KubeCostEnv(task["trace"])
            #     trajectory = agent.run_episode(env, task_name=task["name"])
            #     score = task["grader"].grade(trajectory.steps)
            #     results[task["name"]] = score
            #     print(f"  Score: {score:.3f}")
            # except Exception as e:
            #     print(f"  ERROR: {e}")
            #     results[task["name"]] = 0.0
            
            pass
        
        # Print summary
        print("\n" + "=" * 50)
        print("INFERENCE RESULTS SUMMARY")
        print("=" * 50)
        for task_name, score in results.items():
            print(f"  {task_name}: {score:.3f}")
        
        avg_score = sum(results.values()) / len(results) if results else 0.0
        print(f"\nAverage Score: {avg_score:.3f}")
        print("=" * 50)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
