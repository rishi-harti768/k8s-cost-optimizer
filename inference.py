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

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from models import Action, ActionType, Observation, Trajectory, TrajectoryStep

__all__ = [
    "CostOptimizerAgent",
    "validate_env",
    "main",
]

# Configure module logger
logger = logging.getLogger(__name__)


# ===== CONFIGURATION CONSTANTS =====


class _InferenceConfig:
    """Configuration for inference pipeline."""

    # LLM request parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 100
    LLM_TIMEOUT_SEC: int = 30

    # Episode parameters
    MAX_STEPS_PER_EPISODE: int = 1000

    # Environment variables
    API_BASE_URL_DEFAULT: str = "https://router.huggingface.co/v1"
    MODEL_NAME_DEFAULT: str = "openai/gpt-oss-120b:groq"

    REQUIRED_ENV_VARS: List[str] = ["API_BASE_URL", "MODEL_NAME"]
    REQUIRED_TOKEN_VARS: List[str] = ["HF_TOKEN"]  # HF_TOKEN mandatory for submission

    # Minimum score threshold
    MIN_MODEL_NAME_LENGTH: int = 2
    MIN_API_KEY_LENGTH: int = 10

    # Trace file paths
    TASK_CONFIGS: List[Dict[str, Any]] = [
        {
            "name": "cold_start",
            "trace": "traces/trace_v1_coldstart.json",
            "grader": "ColdStartGrader",
            "description": "Scale cluster from 0→5 replicas without SLA breach",
        },
        {
            "name": "efficient_squeeze",
            "trace": "traces/trace_v1_squeeze.json",
            "grader": "EfficientSqueezeGrader",
            "description": "Maintain <20% steal over 24-hour load cycle",
        },
        {
            "name": "entropy_storm",
            "trace": "traces/trace_v1_entropy.json",
            "grader": "EntropyStormGrader",
            "description": "Proactive REBALANCE_NODE before steal>20%",
        },
    ]

    # Minimum average score to pass
    PASSING_SCORE_THRESHOLD: float = 0.27

    # Output file
    RESULTS_FILE: str = "inference_results.json"
    LOG_FILE: str = "inference.log"


_CONFIG = _InferenceConfig()


# ===== CUSTOM EXCEPTIONS =====


class InferenceError(Exception):
    """Base exception for inference-related errors."""

    pass


class EnvironmentValidationError(InferenceError):
    """Raised when environment variables are invalid."""

    pass


# ===== ENVIRONMENT VARIABLE VALIDATION =====


def validate_env() -> None:
    """
    Validate all required environment variables are set and valid.

    Raises:
        EnvironmentValidationError: If any required env var is missing or invalid.
    """
    # Use defaults when env vars are missing, per challenge requirements
    api_url = os.environ.get("API_BASE_URL", _CONFIG.API_BASE_URL_DEFAULT).strip()
    model_name = os.environ.get("MODEL_NAME", _CONFIG.MODEL_NAME_DEFAULT).strip()

    # HF_TOKEN is mandatory
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        error_msg = "HF_TOKEN environment variable is required"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    # Basic format and sanity validation
    if not (api_url.startswith("http://") or api_url.startswith("https://")):
        error_msg = f"API_BASE_URL must start with http:// or https://: {api_url}"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    if not model_name or len(model_name) < _CONFIG.MIN_MODEL_NAME_LENGTH:
        error_msg = f"MODEL_NAME must be non-empty and at least {_CONFIG.MIN_MODEL_NAME_LENGTH} chars: {model_name}"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    if len(hf_token) < _CONFIG.MIN_API_KEY_LENGTH:
        error_msg = f"HF_TOKEN must have length >= {_CONFIG.MIN_API_KEY_LENGTH}"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    logger.info(f"Using API_BASE_URL={api_url} MODEL_NAME={model_name} HF_TOKEN=****")


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

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAI LLM inference client.

        Args:
            model_name: Model ID (from MODEL_NAME env var if not provided)
            api_base_url: API endpoint (from API_BASE_URL env var if not provided)

        Raises:
            InferenceError: If model_name or api_base_url cannot be determined.

        Spec requirement: Uses OpenAI Client for all LLM calls.
        """
        self.model_name = (model_name or os.environ.get("MODEL_NAME", _CONFIG.MODEL_NAME_DEFAULT)).strip()
        self.api_base_url = (api_base_url or os.environ.get("API_BASE_URL", _CONFIG.API_BASE_URL_DEFAULT)).strip()

        if not self.model_name:
            raise InferenceError("MODEL_NAME env var required")
        if not self.api_base_url:
            raise InferenceError("API_BASE_URL env var required")

        self.api_key = os.environ.get("HF_TOKEN", "").strip()
        if not self.api_key:
            raise InferenceError("HF_TOKEN env var required")

        try:
            # Initialize OpenAI client with API key
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url,
            )
            logger.debug(f"OpenAI client initialized: model={self.model_name}, base_url={self.api_base_url}")
        except Exception as e:
            raise InferenceError(f"Failed to initialize OpenAI client: {e}") from e

    def decide(
        self,
        observation: Observation,
        task_description: str = "",
    ) -> Action:
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
                task_constraints = (
                    "\nPriority: Reach 5+ replicas as quickly as possible "
                    "while keeping p99_latency_ms < 300ms."
                )
            elif task_description.lower() == "efficient_squeeze":
                task_constraints = (
                    "\nPriority: Keep cpu_steal_pct < 20% throughout. "
                    "Balance cost vs. reliability."
                )
            elif task_description.lower() == "entropy_storm":
                task_constraints = (
                    "\nPriority: USE REBALANCE_NODE to prevent cpu_steal_pct from exceeding 20%. "
                    "Be PROACTIVE, not reactive."
                )

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
                    {
                        "role": "system",
                        "content": "You are a Kubernetes cost optimization expert. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=_CONFIG.LLM_TEMPERATURE,
                max_tokens=_CONFIG.LLM_MAX_TOKENS,
                timeout=_CONFIG.LLM_TIMEOUT_SEC,
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
                except KeyError as e:
                    raise ValueError(
                        f"Unknown action: {action_type_str}. "
                        f"Valid: {[a.value for a in ActionType]}"
                    ) from e

            return Action(action_type=action_type)

        except Exception as e:
            logger.warning(f"LLM response parsing failed ({e}), defaulting to MAINTAIN")
            if "response_text" in locals():
                logger.debug(f"Response was: {response_text[:200]}")
            return Action(action_type=ActionType.MAINTAIN)

    def run_episode(
        self,
        env: KubeCostEnv,
        max_steps: int = _CONFIG.MAX_STEPS_PER_EPISODE,
        task_name: str = "",
    ) -> Trajectory:
        """
        Run one full episode with LLM agent.

        Args:
            env: Environment instance (KubeCostEnv)
            max_steps: Max steps before forced termination
            task_name: Task identifier for LLM context

        Returns:
            Trajectory: List of trajectory steps for grading

        Raises:
            InferenceError: If episode produces empty trajectory or other error occurs.
        """
        env_name = getattr(getattr(env, "trace", None), "task_name", "unknown")
        print(f"[START] task={task_name or env_name} env={env_name} model={self.model_name}")

        success = False
        step_rewards: List[float] = []

        try:
            obs = env.reset()

            for step_num in range(max_steps):
                # Get action from LLM
                action = self.decide(obs, task_name)

                step_error: Optional[str] = None
                try:
                    obs, reward, done, info = env.step(action)
                    step_error = info.get("last_action_error") if isinstance(info, dict) else None
                except Exception as exc:
                    reward = 0.0
                    done = True
                    step_error = str(exc)

                step_rewards.append(reward)
                action_text = getattr(action.action_type, "value", str(action.action_type))
                error_text = step_error if step_error is not None else "null"
                print(
                    f"[STEP] step={step_num + 1} action={action_text} "
                    f"reward={reward:.2f} done={str(done).lower()} error={error_text}"
                )

                if done:
                    logger.debug(f"Episode {task_name} terminated at step {step_num + 1}")
                    break

            trajectory = Trajectory(steps=env.trajectory)

            # VALIDATE: trajectory must not be empty
            if not trajectory.steps:
                raise InferenceError(
                    f"Episode produced empty trajectory for task {task_name}"
                )

            success = True
            logger.info(f"Episode {task_name}: {len(trajectory.steps)} steps")
            return trajectory

        except Exception as e:
            logger.error(f"Episode failed for {task_name}: {e}")
            raise

        finally:
            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

            if "trajectory" in locals() and isinstance(trajectory, Trajectory):
                final_steps = len(trajectory.steps)
                rewards_csv = ",".join(f"{s.reward:.2f}" for s in trajectory.steps)
            else:
                final_steps = len(step_rewards)
                rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards)

            print(
                f"[END] success={str(success).lower()} "
                f"steps={final_steps} rewards={rewards_csv}"
            )

    def evaluate_task(
        self,
        env: KubeCostEnv,
        task_name: str,
        grader: Any,
    ) -> float:
        """
        Run episode and score with appropriate grader.

        Args:
            env: Environment instance
            task_name: Task identifier
            grader: Grader instance (ColdStartGrader, etc.)

        Returns:
            float: Score from grader [0.0, 1.0]

        Raises:
            InferenceError: If episode or grading fails.
        """
        try:
            trajectory = self.run_episode(env, task_name=task_name)
            score = grader.grade(trajectory.steps)

            if not (0.0 <= score <= 1.0):
                raise InferenceError(f"Invalid score {score} outside [0.0, 1.0]")

            return score
        except Exception as e:
            logger.error(f"Task evaluation failed for {task_name}: {e}")
            raise


# ===== MAIN INFERENCE ENTRY POINT =====


def main() -> int:
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
        3. Run 3 tasks (cold_start, efficient_squeeze, entropy_storm)
        4. Collect trajectories and scores
        5. Verify all scores in [0.0, 1.0] range
        6. Output results to console/file
        7. Exit with code 0 if passing, else 1

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_CONFIG.LOG_FILE),
        ],
    )

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

        # Build task configs with grader instances
        grader_map = {
            "ColdStartGrader": ColdStartGrader(),
            "EfficientSqueezeGrader": EfficientSqueezeGrader(),
            "EntropyStormGrader": EntropyStormGrader(),
        }

        results: Dict[str, float] = {}

        # Run inference on each task
        for task_config in _CONFIG.TASK_CONFIGS:
            task_name = task_config["name"]
            trace_path = task_config["trace"]
            grader_cls_name = task_config["grader"]
            description = task_config["description"]

            logger.info(f"\n[{task_name.upper()}]")
            logger.info(f"  Description: {description}")

            try:
                env = KubeCostEnv(trace_path)
                grader = grader_map[grader_cls_name]
                score = agent.evaluate_task(env, task_name, grader)

                results[task_name] = score
                logger.info(f"  Score: {score:.3f}")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                results[task_name] = 0.0

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
        with open(_CONFIG.RESULTS_FILE, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "results": results,
                    "average_score": total_score,
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {_CONFIG.RESULTS_FILE}")

        # Determine success
        passing = total_score >= _CONFIG.PASSING_SCORE_THRESHOLD
        logger.info(
            f"\nAverage score {total_score:.3f} "
            f"{'✓ PASSING' if passing else '✗ FAILING'} "
            f"(threshold: {_CONFIG.PASSING_SCORE_THRESHOLD})"
        )

        return 0 if passing else 1

    except EnvironmentValidationError as e:
        logger.error(f"[FATAL] Environment validation failed: {e}")
        return 1
    except InferenceError as e:
        logger.error(f"[FATAL] Inference failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"[FATAL] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
