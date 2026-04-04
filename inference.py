# inference.py
"""
LLM Inference Pipeline for KubeCost-Gym (OpenAI Client).

Location:    ROOT directory (spec §5 requirement).
Runtime:     < 20 minutes on vcpu=2, memory=8gb.

Environment variables (required — all read via os.environ.get):
  API_BASE_URL  : LLM API endpoint   (e.g. https://api.openai.com/v1)
  MODEL_NAME    : Model identifier    (e.g. gpt-4)
  HF_TOKEN      : HuggingFace / API key (injected by validator)

Stdout log format (MANDATORY — evaluated by automated scorer):
  [START] {"task": "<name>", "model": "<model>", "max_steps": <n>}
  [STEP]  {"task": "<name>", "step": <n>, "action": "<action>",
            "reward": <float>, "done": <bool>,
            "obs": { ...observation fields... }}
  [END]   {"task": "<name>", "score": <float>, "total_steps": <n>,
            "status": "success"|"error"}
"""

import json
import logging
import os
import sys
<<<<<<< HEAD
from typing import List, Dict, Any, Optional
=======
from datetime import datetime
from typing import Any, Dict, List, Optional
>>>>>>> main

from openai import OpenAI

from env import KubeCostEnv
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
<<<<<<< HEAD
from models import Observation, Action, ActionType, Trajectory
=======
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
    LLM_MAX_TOKENS: int = 500  # Increased from 100 to accommodate reasoning mode + complete JSON response
    LLM_TIMEOUT_SEC: int = 30

    # Episode parameters
    MAX_STEPS_PER_EPISODE: int = 1000

    # Environment variables
    API_BASE_URL_DEFAULT: str = "https://router.huggingface.co/v1"
    MODEL_NAME_DEFAULT: str = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

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
>>>>>>> main


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

<<<<<<< HEAD
MAX_STEPS_PER_TASK = 200   # Keep well under 20-min ceiling on 2 vCPU

TASKS: List[Dict[str, Any]] = [
    {
        "name":        "cold_start",
        "trace":       "traces/trace_v1_coldstart.json",
        "grader":      ColdStartGrader(),
        "description": "Scale cluster from 0→5 replicas without SLA breach (p99 < 300ms).",
        "difficulty":  "easy",
    },
    {
        "name":        "efficient_squeeze",
        "trace":       "traces/trace_v1_squeeze.json",
        "grader":      EfficientSqueezeGrader(),
        "description": "Maintain cpu_steal_pct < 20% across 24-hour sinusoidal load cycle.",
        "difficulty":  "medium",
    },
    {
        "name":        "entropy_storm",
        "trace":       "traces/trace_v1_entropy.json",
        "grader":      EntropyStormGrader(),
        "description": "Issue REBALANCE_NODE before cpu_steal_pct exceeds 20% (proactive).",
        "difficulty":  "hard",
    },
]
=======

def validate_env() -> None:
    """
    Validate all required environment variables are set and valid at startup.

    Raises:
        EnvironmentValidationError: If any required env var is missing or invalid.
        
    DEPLOYMENT NOTE: For HF Spaces, set these in Space secrets:
      - HF_TOKEN: API key for the LLM endpoint (required, no default)
      - API_BASE_URL: LLM API endpoint (optional, defaults to huggingface router)
      - MODEL_NAME: LLM model ID (optional, defaults to mistral)
    """
    # HF_TOKEN is mandatory - fail fast if missing
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        error_msg = (
            "CRITICAL: HF_TOKEN environment variable is required.\n"
            "          This is the API key for your LLM endpoint.\n"
            "          For HF Spaces, add it to Space secrets."
        )
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)
>>>>>>> main

    if len(hf_token) < _CONFIG.MIN_API_KEY_LENGTH:
        error_msg = f"HF_TOKEN must have length >= {_CONFIG.MIN_API_KEY_LENGTH} characters"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

<<<<<<< HEAD
# ---------------------------------------------------------------------------
# Structured log helpers  (MANDATORY format — do not alter)
# ---------------------------------------------------------------------------

def _log(tag: str, payload: Dict[str, Any]) -> None:
    """Emit a single structured log line to stdout and flush immediately."""
    print(f"{tag} {json.dumps(payload, default=str)}", flush=True)
=======
    # Use defaults for API URL and model name
    api_url = os.environ.get("API_BASE_URL", _CONFIG.API_BASE_URL_DEFAULT).strip()
    model_name = os.environ.get("MODEL_NAME", _CONFIG.MODEL_NAME_DEFAULT).strip()

    # Basic format validation
    if not (api_url.startswith("http://") or api_url.startswith("https://")):
        error_msg = f"API_BASE_URL must start with http:// or https://: {api_url}"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    if not model_name or len(model_name) < _CONFIG.MIN_MODEL_NAME_LENGTH:
        error_msg = f"MODEL_NAME must be non-empty and at least {_CONFIG.MIN_MODEL_NAME_LENGTH} chars: {model_name}"
        logger.error(error_msg)
        raise EnvironmentValidationError(error_msg)

    logger.info(f"✓ Environment variables validated: API_BASE_URL={api_url[:50]}... MODEL_NAME={model_name} HF_TOKEN=****")
>>>>>>> main


def log_start(task_name: str, model: str, max_steps: int) -> None:
    _log("[START]", {"task": task_name, "model": model, "max_steps": max_steps})


def log_step(task_name: str, step: int, action: str,
             reward: float, done: bool, obs: Observation) -> None:
    obs_dict = obs.model_dump()
    # Ensure node_size_class is a plain string
    nsc = obs_dict.get("node_size_class")
    obs_dict["node_size_class"] = nsc.value if hasattr(nsc, "value") else str(nsc)
    _log("[STEP]", {
        "task":   task_name,
        "step":   step,
        "action": action,
        "reward": round(float(reward), 4),
        "done":   bool(done),
        "obs":    obs_dict,
    })


def log_end(task_name: str, score: float, total_steps: int, status: str) -> None:
    _log("[END]", {
        "task":        task_name,
        "score":       round(float(score), 4),
        "total_steps": total_steps,
        "status":      status,
    })


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------


class CostOptimizerAgent:
    """
<<<<<<< HEAD
    LLM-based agent that queries an OpenAI-compatible API for actions.

    Uses os.environ.get() for all credentials — no hardcoded values.
    """

    SYSTEM_PROMPT = (
        "You are a Kubernetes cost optimization expert. "
        "Analyse the cluster state and return ONLY a JSON object with one field: "
        "action_type. Choose from the available actions list provided."
    )

    def __init__(self) -> None:
        self.model_name:   str = os.environ.get("MODEL_NAME", "")
        self.api_base_url: str = os.environ.get("API_BASE_URL", "")
        self.hf_token:     str = os.environ.get("HF_TOKEN", "")

        if not self.model_name:
            raise ValueError("MODEL_NAME environment variable is not set.")
        if not self.api_base_url:
            raise ValueError("API_BASE_URL environment variable is not set.")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is not set.")

        # OpenAI client — spec mandates OpenAI client for all LLM calls
        self.client = OpenAI(
            api_key=self.hf_token,
            base_url=self.api_base_url,
        )

    # ------------------------------------------------------------------

    def decide(self, obs: Observation, task_description: str = "") -> Action:
        """
        Query LLM and parse its chosen action.

        Falls back to MAINTAIN on any error so the episode can continue.
=======
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
>>>>>>> main
        """
        available_actions = ", ".join(a.value for a in ActionType)
        obs_json = json.dumps(obs.model_dump(), default=str, indent=2)

        user_prompt = (
            f"Task: {task_description}\n\n"
            f"Available actions: {available_actions}\n\n"
            f"Current cluster state:\n{obs_json}\n\n"
            'Respond with ONLY valid JSON, e.g. {"action_type": "MAINTAIN"}'
        )

        try:
<<<<<<< HEAD
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",  "content": self.SYSTEM_PROMPT},
                    {"role": "user",    "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()

            # Strip optional markdown fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)
            action_type = ActionType(data["action_type"])
            return Action(action_type=action_type)

        except Exception as exc:
            print(f"[WARN] LLM decision failed ({exc}), defaulting to MAINTAIN",
                  file=sys.stderr, flush=True)
            return Action(action_type=ActionType.MAINTAIN)

    # ------------------------------------------------------------------

    def run_task(self, task: Dict[str, Any]) -> float:
        """
        Run a full episode for one task, emit structured logs, return score.

        Emits:
            [START] once at beginning
            [STEP]  every environment step
            [END]   once at completion
        """
        task_name   = task["name"]
        description = task["description"]
        grader      = task["grader"]
        trace_path  = task["trace"]

        log_start(task_name, self.model_name, MAX_STEPS_PER_TASK)

        total_steps = 0
        score       = 0.0
        status      = "success"
=======
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

            # Log raw API response for debugging
            logger.debug(f"[API Response] status: {response.model_dump() if hasattr(response, 'model_dump') else response}")
            logger.debug(f"[Choices] count={len(response.choices) if response.choices else 0}")
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                logger.debug(f"[Choice 0] message={choice.message}, finish_reason={choice.finish_reason}")
                if choice.message:
                    logger.debug(f"[Message] role={choice.message.role}, content={repr(choice.message.content)}")

            # Validate response structure
            if not response.choices or len(response.choices) == 0:
                raise ValueError("API returned no choices in response")
            
            if response.choices[0].message is None:
                raise ValueError("API returned None message in first choice")

            # Parse response (handle markdown code blocks)
            response_content = response.choices[0].message.content
            
            # FALLBACK: Use reasoning field if content is None (reasoning mode / token limit case)
            if response_content is None:
                logger.warning(f"Content is None. Model may be in reasoning mode. Checking reasoning field...")
                if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                    reasoning_text = response.choices[0].message.reasoning
                    logger.debug(f"[Reasoning Fallback] Attempting to extract action from reasoning: {reasoning_text[:200]}...")
                    
                    # Try to extract action from reasoning (heuristic approach)
                    # Look for action names in the reasoning text
                    for action in ActionType:
                        if action.value in reasoning_text:
                            logger.info(f"[Fallback Success] Extracted action from reasoning: {action.value}")
                            return Action(action_type=action)
                
                logger.error(f"[ERROR] API returned None content. Reasoning also empty/no action found. Full response: {response.model_dump() if hasattr(response, 'model_dump') else response}")
                raise ValueError("LLM returned empty response (content is None and no action found in reasoning)")
            
            response_text = response_content.strip()
            logger.debug(f"[Response Text] {response_text[:500]}")

            # Extract JSON (handle markdown code blocks)
            if "```" in response_text:
                json_str = response_text.split("```")[1].strip()
                if json_str.startswith("json"):
                    json_str = json_str[4:].strip()
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
                logger.debug(f"Response text: {response_text[:300]}")
            if "response" in locals():
                try:
                    logger.debug(f"Full API response dump: {response.model_dump() if hasattr(response, 'model_dump') else str(response)}")
                except Exception as dump_err:
                    logger.debug(f"Could not dump response: {dump_err}")
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
>>>>>>> main

        try:
            env = KubeCostEnv(trace_path)
            obs = env.reset()

<<<<<<< HEAD
            for step_num in range(1, MAX_STEPS_PER_TASK + 1):
                action = self.decide(obs, description)
                obs, reward, done, _info = env.step(action)
                total_steps = step_num

                log_step(
                    task_name=task_name,
                    step=step_num,
                    action=action.action_type.value,
                    reward=reward,
                    done=done,
                    obs=obs,
=======
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
>>>>>>> main
                )

                if done:
                    logger.debug(f"Episode {task_name} terminated at step {step_num + 1}")
                    break

<<<<<<< HEAD
            # Grade the completed trajectory
            trajectory = env.trajectory
            score = grader.grade(trajectory)
            score = max(0.0, min(1.0, score))   # Hard clamp — spec §4

        except Exception as exc:
            print(f"[ERROR] Task '{task_name}' failed: {exc}",
                  file=sys.stderr, flush=True)
            status = "error"
            score  = 0.0

        log_end(task_name, score, total_steps, status)
        return score
=======
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
>>>>>>> main


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

<<<<<<< HEAD
def main() -> None:
    """
    Run all three tasks sequentially and print a final summary.

    Exit codes:
        0 — all tasks completed (scores may vary)
        1 — fatal startup error (missing env vars, import failure, etc.)
    """

    # ---- Validate required env vars ----------------------------------------
    missing = [k for k in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
               if not os.environ.get(k)]
    if missing:
        print(f"[ERROR] Missing required environment variables: {', '.join(missing)}",
              file=sys.stderr, flush=True)
        sys.exit(1)
=======

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
    # Setup logging (DEBUG level to capture detailed response logs)
    logging.basicConfig(
        level=logging.DEBUG,
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
>>>>>>> main

    print(f"[INFO] API_BASE_URL : {os.environ.get('API_BASE_URL')}", flush=True)
    print(f"[INFO] MODEL_NAME   : {os.environ.get('MODEL_NAME')}", flush=True)
    print(f"[INFO] HF_TOKEN     : {'*' * 8} (hidden)", flush=True)

    # ---- Initialise agent --------------------------------------------------
    try:
        agent = CostOptimizerAgent()
    except Exception as exc:
        print(f"[ERROR] Agent init failed: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    # ---- Run all tasks -----------------------------------------------------
    results: Dict[str, float] = {}

    for task in TASKS:
        score = agent.run_task(task)
        results[task["name"]] = score

    # ---- Final summary (plain text, for human readers) --------------------
    print("\n" + "=" * 60, flush=True)
    print("INFERENCE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task_name, score in results.items():
        flag = "PASS" if 0.0 <= score <= 1.0 else "FAIL"
        print(f"  [{flag}] {task_name}: {score:.4f}", flush=True)

    avg = sum(results.values()) / len(results) if results else 0.0
    print(f"\n  Average score : {avg:.4f}", flush=True)
    print("=" * 60, flush=True)

    sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
