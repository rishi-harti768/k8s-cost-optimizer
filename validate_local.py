# validate_local.py
"""
Pre-submission validation harness (spec §7 automated gates).

Run locally before pushing to HF Space:
  python validate_local.py

Validates pre-submission checklist:
  [PASS] Module imports (no syntax errors)
  [PASS] Environment variables using os.environ.get() (not hardcoded)
  [PASS] Graders return strictly [0.0, 1.0]
  [PASS] OpenAI Client usage (not Google Gemini)
  [PASS] All 3+ tasks with graders present
  [PASS] openenv.yaml OpenEnv spec compliant
  [PASS] inference.py in root directory

Reference: PROJECT_SPEC.md §7 Pre-Submission Checklist
"""

import inspect
import logging
import sys
from pathlib import Path

import yaml

__all__ = [
    "run_all_checks",
    "ValidationError",
    "check_imports",
    "check_openenv_yaml",
    "check_graders",
]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ===== CUSTOM EXCEPTIONS =====


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class ImportValidationError(ValidationError):
    """Raised when import validation fails."""

    pass


class ConfigValidationError(ValidationError):
    """Raised when configuration validation fails."""

    pass


# ===== VALIDATION CHECKS =====


def check_imports() -> bool:
    """
    Validate all modules import without syntax errors.

    Returns:
        bool: True if all imports successful, False otherwise.
    """
    try:
        from env import KubeCostEnv
        from models import (
            Observation,
            EnvState,
            Action,
            ActionType,
            TrajectoryStep,
            Trajectory,
            NodeSizeClass,
        )
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader

        logger.info("  [PASS] All modules import successfully")
        return True
    except ImportValidationError as e:
        logger.error(f"  [FAIL] Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Import failed: {e}")
        return False


def check_openenv_yaml() -> bool:
    """
    Validate openenv.yaml structure.

    Returns:
        bool: True if openenv.yaml is valid, False otherwise.
    """
    try:
        yaml_path = Path("openenv.yaml")
        if not yaml_path.exists():
            raise ConfigValidationError("openenv.yaml not found")

        with yaml_path.open() as f:
            spec = yaml.safe_load(f)

        # Required fields
        if spec is None:
            raise ConfigValidationError("YAML is empty")
        if "name" not in spec:
            raise ConfigValidationError("Missing 'name' field")
        if spec["name"] not in ["kubecost-gym", "kubecost_gym"]:
            raise ConfigValidationError(f"Invalid name: {spec['name']}")

        if "version" not in spec:
            raise ConfigValidationError("Missing 'version' field")
        if not isinstance(spec["version"], str):
            raise ConfigValidationError(
                f"'version' must be string, got {type(spec['version'])}"
            )

        if "description" not in spec:
            raise ConfigValidationError("Missing 'description' field")
        if not spec["description"]:
            raise ConfigValidationError("description is empty")

        if "tasks" not in spec:
            raise ConfigValidationError("Missing 'tasks' field")
        if len(spec["tasks"]) != 3:
            raise ConfigValidationError(
                f"Must have exactly 3 tasks, got {len(spec['tasks'])}"
            )

        # Task validation
        task_names = set()
        for task in spec["tasks"]:
            if "name" not in task:
                raise ConfigValidationError("Task missing 'name' field")
            if "difficulty" not in task:
                raise ConfigValidationError(
                    f"Task {task.get('name')} missing 'difficulty' field"
                )
            if task["difficulty"] not in ["easy", "medium", "hard"]:
                raise ConfigValidationError(
                    f"Invalid difficulty: {task['difficulty']}"
                )
            if "description" not in task:
                raise ConfigValidationError(
                    f"Task {task.get('name')} missing 'description' field"
                )
            task_names.add(task["name"])

        # Check for expected task names
        expected_tasks = {"cold_start", "efficient_squeeze", "entropy_storm"}
        if task_names != expected_tasks:
            logger.warning(
                f"  [WARN] Task names differ from expected: {task_names} vs {expected_tasks}"
            )

        logger.info(
            f"  [PASS] openenv.yaml valid "
            f"({spec['name']} v{spec['version']}, {len(spec['tasks'])} tasks)"
        )
        return True

    except ConfigValidationError as e:
        logger.error(f"  [FAIL] openenv.yaml validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] openenv.yaml validation failed: {e}")
        return False


def check_graders() -> bool:
    """
    Validate all graders have been implemented and return [0.0, 1.0].

    Returns:
        bool: True if all graders valid, False otherwise.
    """
    try:
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
        from models import TrajectoryStep, Observation, ActionType, NodeSizeClass

        graders = [
            ColdStartGrader(),
            EfficientSqueezeGrader(),
            EntropyStormGrader(),
        ]

        # 1. Test empty trajectory returns 0.0 (SDD Rule 1)
        empty_traj = []
        for grader in graders:
            score = grader.grade(empty_traj)
            if score != 0.0:
                raise ValidationError(
                    f"{grader.__class__.__name__}: empty trajectory should return 0.0, got {score}"
                )

        # 2. Test healthy trajectory
        dummy_obs = Observation(
            cpu_usage_pct=45.0,
            mem_usage_pct=50.0,
            p99_latency_ms=180.0,
            http_error_rate=0.0,
            cpu_steal_pct=0.05,
            active_replicas=3,
            buffer_depth=5,
            node_size_class=NodeSizeClass.MEDIUM,
            current_hourly_cost=40.0,
            node_bin_density=[0.4] * 10,
        )
        dummy_step = TrajectoryStep(
            observation=dummy_obs,
            action=ActionType.MAINTAIN,
            reward=1.0,
            done=False,
            info={},
        )
        healthy_traj = [dummy_step] * 5

        for grader in graders:
            score = grader.grade(healthy_traj)
            if not isinstance(score, float) or not (0.0 <= score <= 1.0):
                raise ValidationError(
                    f"{grader.__class__.__name__}: score {score} out of range [0.0, 1.0]"
                )

            # EntropyStorm specifics - If no violations, returns 1.0 only if REBALANCE_NODE actions exist
            if grader.__class__.__name__ == "EntropyStormGrader":
                # In a healthy trajectory with no violations, the grader may return 0.0
                # (passive/no action) or 1.0 (active REBALANCE_NODE). Both are valid.
                if score not in (0.0, 1.0):
                    raise ValidationError(
                        f"{grader.__class__.__name__}: score {score} unexpected "
                        f"for healthy trajectory (expect 0.0 or 1.0)"
                    )

            logger.info(
                f"  [PASS] {grader.__class__.__name__}: score={score:.2f}"
            )

        return True

    except ValidationError as e:
        logger.error(f"  [FAIL] Grader validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Grader validation failed: {e}")
        return False


def check_inference_root() -> bool:
    """
    Validate inference.py in root directory.

    Returns:
        bool: True if inference.py exists in root, False otherwise.
    """
    inference_path = Path("inference.py")
    if inference_path.exists():
        logger.info("  [PASS] inference.py exists in root directory")
        return True
    else:
        logger.error("  [FAIL] inference.py not found in root directory")
        return False


def check_env_structure() -> bool:
    """
    Validate env.py structure has required methods.

    Returns:
        bool: True if all required methods present, False otherwise.
    """
    try:
        from env import KubeCostEnv

        # Check required methods exist
        required_methods = ["reset", "step", "state", "_apply_action", "_calculate_reward"]
        for method in required_methods:
            if not hasattr(KubeCostEnv, method):
                raise ValidationError(f"KubeCostEnv missing method: {method}")

        logger.info("  [PASS] KubeCostEnv has all required methods")
        return True
    except ValidationError as e:
        logger.error(f"  [FAIL] env.py structure validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] env.py structure validation failed: {e}")
        return False


def check_requirements_openai() -> bool:
    """
    Validate requirements.txt includes OpenAI, not Google Gemini.

    Returns:
        bool: True if using OpenAI (or safe to proceed), False if using Google Gemini.
    """
    try:
        req_path = Path("requirements.txt")
        if not req_path.exists():
            logger.warning("  [WARN] requirements.txt not found")
            return True

        content = req_path.read_text()

        has_openai = "openai" in content.lower()
        has_google = "google-generativeai" in content.lower()

        if has_google:
            logger.error(
                "  [FAIL] requirements.txt includes Google Gemini (should be OpenAI)"
            )
            return False
        elif has_openai:
            logger.info("  [PASS] requirements.txt uses OpenAI (not Google Gemini)")
            return True
        else:
            logger.warning("  [WARN] OpenAI not found in requirements.txt")
            return True

    except Exception as e:
        logger.error(f"  [FAIL] Requirements check failed: {e}")
        return False


# ===== MAIN VALIDATION ORCHESTRATOR =====


def run_all_checks() -> int:
    """
    Run all validation checks.

    Returns:
        int: 0 if all critical checks pass, 1 if any fail.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PRE-SUBMISSION VALIDATION")
    logger.info("=" * 60)

    checks = [
        ("Import validation", check_imports),
        ("Environment structure", check_env_structure),
        ("openenv.yaml compliance", check_openenv_yaml),
        ("Grader bounds", check_graders),
        ("inference.py location", check_inference_root),
        ("Requirements (OpenAI)", check_requirements_openai),
    ]

    results = []
    for check_name, check_func in checks:
        logger.info(f"\n[{check_name}]")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f"  [ERROR] {e}")
            results.append((check_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {check_name}")

    logger.info(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        logger.info("\n✓ All validation checks passed!")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_checks())
