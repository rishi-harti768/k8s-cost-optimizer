# inference.py
"""
LLM Inference Pipeline for KubeCost-Gym (OpenAI Client).

Location: ROOT directory (spec §5 requirement).

Environment variables (required):
  - API_BASE_URL: The API endpoint for the LLM (e.g., https://api.openai.com/v1)
  - MODEL_NAME: The model identifier to use for inference (e.g., "gpt-4")
  - HF_TOKEN: Hugging Face API token for Space submission (os.environ.get used)

Runtime requirement: Complete end-to-end in <20 minutes on vcpu=2, memory=8gb.

Reference: PROJECT_SPEC.md §5 Infra Spec, §7 Inference Contract
"""

import os
import json
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from models import Observation, Action, ActionType, TrajectoryStep, Trajectory

# Module-level logger (Fix #23)
logger = logging.getLogger(__name__)

# ===== ENVIRONMENT VARIABLE VALIDATION =====

def get_env_or_raise(key: str, default: str = None) -> str:
    """
    Get environment variable or raise if missing and no default.
    
    CRITICAL: Uses os.environ.get() as spec-required (not hardcoded strings).
    
    Args:
        key: Environment variable name
        default: Default value if not found (optional)
    
    Returns:
        str: Environment variable value
    
    Raises:
        ValueError: If key not found and no default provided
    """
    value = os.environ.get(key, default)
    if not value:
        raise ValueError(f"Missing required env var: {key}")
    return value


def validate_env():
    """Validate all required environment variables are set and valid."""
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")
    
    # Format validation
    api_url = os.environ.get("API_BASE_URL").strip()
    if not (api_url.startswith("http://") or api_url.startswith("https://")):
        raise ValueError(f"API_BASE_URL must start with http:// or https://: {api_url}")
    
    model_name = os.environ.get("MODEL_NAME").strip()
    if not model_name or len(model_name) < 2:
        raise ValueError(f"MODEL_NAME must be non-empty: {model_name}")
    
    hf_token = os.environ.get("HF_TOKEN").strip()
    if len(hf_token) < 10:
        raise ValueError(f"HF_TOKEN appears invalid (too short)")


# ===== INFERENCE PIPELINE =====

class CostOptimizerAgent:
    """
    LLM-based decision agent for cost optimization (OpenAI Client).
    
    Responsibilities:
        - Observe environment state (Observation model)
        - Query LLM for action recommendation (JSON response)
        - Validate response, extract ActionType
        - Execute step, collect trajectory
        - Score trajectory with graders
    """
    
    def __init__(self, model_name: str | None = None, api_base_url: str | None = None) -> None:
        """
        Initialize OpenAI LLM inference client.
        
        Args:
            model_name: Model ID (from MODEL_NAME env var if not provided)
            api_base_url: API endpoint (from API_BASE_URL env var if not provided)
        
        Spec requirement: Uses OpenAI Client for all LLM calls.
        """
        self.model_name = model_name or os.environ.get("MODEL_NAME")
        self.api_base_url = api_base_url or os.environ.get("API_BASE_URL")
        
        if not self.model_name:
            raise ValueError("MODEL_NAME env var required")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL env var required")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.environ.get("HF_TOKEN"),  # Validator will inject HF_TOKEN
            base_url=self.api_base_url
        )
    
    def decide(self, observation: Observation, task_description: str = "") -> Action:
        """
        Query LLM for action given current observation.
        
        Args:
            observation: Current Observation model
            task_description: Task context for LLM reasoning
        
        Returns:
            Action: Selected action with validation
        
        LLM is prompted to:
            - Analyze cluster state
            - Select action from available ActionType enum
            - Return JSON with action_type field
        
        Error handling:
            - If LLM response invalid: default to MAINTAIN
            - Log failures for debugging
        """
        try:
            # Serialize observation to JSON for context
            obs_json = json.dumps(observation.model_dump(), indent=2)
            
            # Build task-specific constraints
            task_constraints = ""
            if task_description.lower() == "cold_start":
                task_constraints = "\nPriority: Reach 5+ replicas as quickly as possible while keeping p99_latency_ms < 300ms."
            elif task_description.lower() == "efficient_squeeze":
                task_constraints = "\nPriority: Keep cpu_steal_pct < 20% throughout. Balance cost vs. reliability."
            elif task_description.lower() == "entropy_storm":
                task_constraints = "\nPriority: USE REBALANCE_NODE to prevent cpu_steal_pct from exceeding 20%. Be PROACTIVE, not reactive."
            
            # Construct prompt
            prompt = f"""Analyze this Kubernetes cluster state and decide on a cost optimization action.

Task: {task_description or "General cost optimization"}{task_constraints}

Current Cluster State:
{obs_json}

SLA Targets:
- p99_latency_ms: < 300ms (healthy), < 200ms (optimal)
- http_error_rate: < 0.01 (< 1%)
- cpu_steal_pct: < 0.20 (< 20%, leading indicator of problems)

Available actions:
{', '.join([action.value for action in ActionType])}

Respond with ONLY valid JSON (no markdown):
{{"action_type": "<one of the above actions>"}}"""
            
            # Query LLM with timeout
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Kubernetes cost optimization expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100,
                timeout=30  # 30-second timeout per request
            )
            
            # Parse response (handle markdown code blocks)
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON (handle markdown code blocks)
            if "```" in response_text:
                json_str = response_text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                response_json = json.loads(json_str)
            else:
                response_json = json.loads(response_text)
            
            # Validate required field
            if "action_type" not in response_json:
                raise ValueError("Response missing 'action_type' field")
            
            action_type_str = response_json["action_type"]
            
            # Validate action exists in enum
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                # Try matching by name instead of value
                try:
                    action_type = ActionType[action_type_str]
                except KeyError:
                    raise ValueError(f"Unknown action: {action_type_str}. Valid: {[a.value for a in ActionType]}")
            
            return Action(action_type=action_type)
            
        except Exception as e:
            logger.warning(f"LLM response parsing failed ({e}), defaulting to MAINTAIN")
            if 'response_text' in locals():
                logger.debug(f"Response was: {response_text[:200]}")
            return Action(action_type=ActionType.MAINTAIN)
    
    def run_episode(self, env: KubeCostEnv, max_steps: int = 1000, task_name: str = "") -> Trajectory:
        """
        Run one full episode with LLM agent.
        
        Args:
            env: Environment instance (KubeCostEnv)
            max_steps: Max steps before forced termination
            task_name: Task identifier for LLM context
        
        Returns:
            Trajectory: List of trajectory steps for grading
            
        Raises:
            ValueError: If episode produces empty trajectory
        """
        try:
            obs = env.reset()
            
            for step_num in range(max_steps):
                # Get action from LLM
                action = self.decide(obs, task_name)
                
                # Execute step in environment (env logs trajectory internally)
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
            
            trajectory = Trajectory(steps=env.trajectory)
            
            # VALIDATE: trajectory must not be empty
            if not trajectory.steps:
                raise ValueError(f"Episode produced empty trajectory for task {task_name}")
            
            return trajectory
            
        except Exception as e:
            logger.error(f"Episode failed for {task_name}: {e}")
            raise
    
    def evaluate_task(self, env: KubeCostEnv, task_name: str, grader) -> float:
        """
        Run episode and score with appropriate grader.
        
        Args:
            env: Environment instance
            task_name: Task identifier
            grader: Grader instance
        
        Returns:
            float: Score from grader [0.0, 1.0]
        """
        trajectory = self.run_episode(env, task_name=task_name)
        score = grader.grade(trajectory.steps)
        return score


# ===== MAIN INFERENCE ENTRY POINT =====

def main():
    """
    Main inference entry point.
    
    Spec requirements (PROJECT_SPEC.md §5 Infra Spec):
        - Runs when executed: python inference.py
        - Must complete in <20 minutes on vcpu=2, memory=8gb
        - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
        - Uses OpenAI Client for all LLM calls
        - Outputs results (trajectories, scores)
    
    Workflow:
        1. Validate environment variables
        2. Initialize agent and environments
        3. Run 3 tasks:
           - cold_start.json with ColdStartGrader
           - efficient_squeeze.json with EfficientSqueezeGrader
           - entropy_storm.json with EntropyStormGrader
        4. Collect trajectories and scores
        5. Verify all scores in [0.0, 1.0] range
        6. Output results to console/file
        7. Exit successfully (code 0) if all scores valid, else exit 1
    """
    # Setup logging (Fix #23)
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('inference.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment
        validate_env()
        logger.info("✓ Environment variables validated")
        logger.info(f"  - API_BASE_URL: {os.environ.get('API_BASE_URL')}")
        logger.info(f"  - MODEL_NAME: {os.environ.get('MODEL_NAME')}")
        logger.info(f"  - HF_TOKEN: {'*' * 4} (hidden)")
        
        # Initialize agent
        model_name = os.environ.get("MODEL_NAME")
        api_base_url = os.environ.get("API_BASE_URL")
        agent = CostOptimizerAgent(model_name=model_name, api_base_url=api_base_url)
        logger.info(f"✓ Agent initialized with model: {model_name}")
        
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
            logger.info(f"\n[{task['name'].upper()}]")
            logger.info(f"  Description: {task['description']}")
            
            try:
                env = KubeCostEnv(task["trace"])
                trajectory = agent.run_episode(env, task_name=task["name"])
                score = task["grader"].grade(trajectory.steps)
                
                # Validate score in [0.0, 1.0]
                if not (0.0 <= score <= 1.0):
                    raise ValueError(f"Score {score} outside bounds [0.0, 1.0]")
                
                results[task["name"]] = score
                logger.info(f"  Score: {score:.3f}")
            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results[task["name"]] = 0.0
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("INFERENCE RESULTS SUMMARY")
        logger.info("=" * 50)
        
        total_score = sum(results.values()) / len(results) if results else 0.0
        logger.info(f"\nTask Scores:")
        for task_name, score in sorted(results.items()):
            status = "✓" if score > 0.5 else "✗"
            logger.info(f"  {status} {task_name}: {score:.3f}")
        
        logger.info(f"\nAverage Score: {total_score:.3f}")
        logger.info("=" * 50)
        
        # Validate all scores in valid range
        invalid_scores = [s for s in results.values() if not (0.0 <= s <= 1.0)]
        if invalid_scores:
            logger.error(f"Invalid scores detected: {invalid_scores}")
            return 1
        
        # Write results to file
        results_file = "inference_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "average_score": total_score
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        return 0 if total_score >= 0.27 else 1
        
    except Exception as e:
        logger.error(f"[FATAL] Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
