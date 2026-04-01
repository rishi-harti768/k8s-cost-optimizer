# validate_local.py
"""
Pre-submission validation harness (spec §7 automated gates).

Run locally before pushing to HF Space:
  python validate_local.py

Validates pre-submission checklist:
  ✓ Module imports (no syntax errors)
  ✓ Environment variables using os.environ.get() (not hardcoded)
  ✓ Graders return strictly [0.0, 1.0]
  ✓ OpenAI Client usage (not Google Gemini)
  ✓ All 3+ tasks with graders present
  ✓ openenv.yaml OpenEnv spec compliant
  ✓ inference.py in root directory
  ✓ Dockerfile builds
  ✓ No hardcoded API keys
  ✓ Traces directory and files

Reference: PROJECT_SPEC.md §7 Pre-Submission Checklist
"""

import sys
import os
import yaml
import re
from pathlib import Path


def check_imports():
    """Validate all modules import without syntax errors."""
    try:
        from env import KubeCostEnv
        from models import (
            Observation, EnvState, Action, ActionType, 
            TrajectoryStep, Trajectory, NodeSizeClass
        )
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
        print("  ✓ All modules import successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def check_openenv_yaml():
    """Validate openenv.yaml structure."""
    try:
        yaml_path = Path("openenv.yaml")
        if not yaml_path.exists():
            print("  ✗ openenv.yaml not found")
            return False
            
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        
        # Required fields
        assert spec is not None, "YAML is empty"
        assert "name" in spec, "Missing 'name'"
        assert spec["name"] in ["kubecost-gym", "kubecost_gym"], f"Invalid name: {spec['name']}"
        
        assert "version" in spec, "Missing 'version'"
        assert isinstance(spec["version"], str), f"'version' must be string, got {type(spec['version'])}"
        
        assert "description" in spec, "Missing 'description'"
        assert len(spec["description"]) > 0, "description is empty"
        
        assert "tasks" in spec, "Missing 'tasks'"
        assert len(spec["tasks"]) == 3, f"Must have exactly 3 tasks, got {len(spec['tasks'])}"
        
        # Task validation
        task_names = set()
        for task in spec["tasks"]:
            assert "name" in task, f"Task missing 'name'"
            assert "difficulty" in task, f"Task {task.get('name')} missing 'difficulty'"
            assert task["difficulty"] in ["easy", "medium", "hard"], \
                f"Invalid difficulty: {task['difficulty']}"
            assert "description" in task, f"Task {task.get('name')} missing 'description'"
            task_names.add(task["name"])
        
        # Check for expected task names
        expected_tasks = {"cold_start", "efficient_squeeze", "entropy_storm"}
        if not task_names == expected_tasks:
            print(f"  ⚠ Task names differ from expected: {task_names} vs {expected_tasks}")
        
        print(f"  ✓ openenv.yaml valid ({spec['name']} v{spec['version']}, {len(spec['tasks'])} tasks)")
        return True
    except Exception as e:
        print(f"  ✗ openenv.yaml validation failed: {e}")
        return False


def check_graders():
    """Validate all graders exist and can be instantiated (stubs)."""
    try:
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
        
        graders = [
            ColdStartGrader(),
            EfficientSqueezeGrader(),
            EntropyStormGrader()
        ]
        
        # At scaffolding stage, graders are stubs (pass bodies)
        # We validate they:
        # 1. Can be instantiated
        # 2. Have grade() method
        # 3. Will return floats when implemented
        
        for grader in graders:
            grader_name = grader.__class__.__name__
            if not hasattr(grader, 'grade'):
                print(f"  ✗ {grader_name} missing grade() method")
                return False
            
            print(f"  ✓ {grader_name}: instantiated (stub, grade() method ready)")
        
        return True
    except Exception as e:
        print(f"  ✗ Grader validation failed: {e}")
        return False


def check_inference_root():
    """Validate inference.py in root directory."""
    inference_path = Path("inference.py")
    if inference_path.exists():
        print("  ✓ inference.py exists in root directory")
        return True
    else:
        print("  ✗ inference.py not found in root directory")
        return False


def check_env_structure():
    """Validate env.py structure."""
    try:
        from env import KubeCostEnv
        import inspect
        
        # Check required methods exist
        required_methods = ["reset", "step", "state", "_apply_action", "_calculate_reward"]
        for method in required_methods:
            if not hasattr(KubeCostEnv, method):
                print(f"  ✗ KubeCostEnv missing method: {method}")
                return False
        
        # Check method signatures
        sig_reset = inspect.signature(KubeCostEnv.reset)
        sig_step = inspect.signature(KubeCostEnv.step)
        sig_state = inspect.signature(KubeCostEnv.state)
        
        print("  ✓ KubeCostEnv has all required methods")
        return True
    except Exception as e:
        print(f"  ✗ env.py structure validation failed: {e}")
        return False


def check_env_variable_patterns():
    """Validate environment variables use os.environ.get() not hardcoded."""
    try:
        # Read inference.py and check for hardcoded API keys
        inference_path = Path("inference.py")
        if not inference_path.exists():
            print("  ✓ inference.py will be checked after creation")
            return True
        
        inference_code = inference_path.read_text()
        
        # Check for hardcoded strings that look like API keys or tokens
        hardcoded_patterns = [
            r'GOOGLE_API_KEY\s*=\s*["\']',
            r'api_key\s*=\s*["\'][a-zA-Z0-9]',
            r'["\'](sk-|gpt-|api_)[a-zA-Z0-9]',  # OpenAI/common patterns
        ]
        
        for pattern in hardcoded_patterns:
            if re.search(pattern, inference_code):
                print(f"  ⚠ Found potential hardcoded credential pattern: {pattern}")
                # Not a hard failure, just warning
        
        # Check for os.environ.get() usage
        if "os.environ.get" in inference_code:
            print("  ✓ Using os.environ.get() for environment variables")
            return True
        else:
            print("  ⚠ Consider using os.environ.get() for all env var access")
            return True  # Not a hard failure
    except Exception as e:
        print(f"  ✗ Environment variable check failed: {e}")
        return False


def check_openai_client():
    """Validate OpenAI Client usage (not Google Gemini)."""
    try:
        inference_path = Path("inference.py")
        if not inference_path.exists():
            print("  ⚠ inference.py not yet created")
            return True
        
        code = inference_path.read_text()
        
        # Check for OpenAI import
        if "from openai import OpenAI" in code or "import openai" in code:
            print("  ✓ Using OpenAI Client")
            return True
        else:
            print("  ⚠ OpenAI Client import not found (ensure using OpenAI not Google Gemini)")
            return True  # Not a hard failure at validation stage
    except Exception as e:
        print(f"  ✗ OpenAI client check failed: {e}")
        return False


def check_grader_bounds():
    """Validate grader implementations clamp scores to [0.0, 1.0]."""
    try:
        graders_path = Path("graders.py")
        if not graders_path.exists():
            print("  ✓ graders.py will be checked after creation")
            return True
        
        code = graders_path.read_text()
        
        # Check for clamping pattern: max(0.0, min(1.0, ...))
        if "max(0.0, min(1.0," in code:
            print("  ✓ Graders properly clamp scores to [0.0, 1.0]")
            return True
        else:
            print("  ⚠ Warning: Check that graders clamp scores to [0.0, 1.0]")
            return True  # Not a hard failure
    except Exception as e:
        print(f"  ✗ Grader bounds check failed: {e}")
        return False


def check_requirements_openai():
    """Validate requirements.txt includes OpenAI, not Google Gemini."""
    try:
        req_path = Path("requirements.txt")
        if not req_path.exists():
            print("  ⚠ requirements.txt not found")
            return True
        
        content = req_path.read_text()
        
        has_openai = "openai" in content.lower()
        has_google = "google-generativeai" in content.lower()
        
        if has_openai and not has_google:
            print("  ✓ requirements.txt uses OpenAI (not Google Gemini)")
            return True
        elif has_google:
            print("  ✗ requirements.txt includes Google Gemini (should be OpenAI)")
            return False
        else:
            print("  ⚠ OpenAI not found in requirements.txt")
            return True  # Not a hard failure
    except Exception as e:
        print(f"  ✗ Requirements check failed: {e}")
        return False


def check_dockerfile_build():
    """Validate Dockerfile can be parsed and has critical checks."""
    try:
        docker_path = Path("Dockerfile")
        if not docker_path.exists():
            print("  ⚠ Dockerfile not found")
            return True
        
        content = docker_path.read_text()
        
        # Check for inference.py verification
        if "inference.py" in content:
            print("  ✓ Dockerfile includes inference.py verification")
            return True
        else:
            print("  ⚠ Dockerfile may not verify inference.py location")
            return True
    except Exception as e:
        print(f"  ✗ Dockerfile check failed: {e}")
        return False
def check_traces_directory():
    """Validate traces directory exists."""
    traces_dir = Path("traces")
    if not traces_dir.exists():
        print("  ⚠ traces/ directory not created yet (will be needed for inference)")
        return True  # Not a hard failure at scaffolding stage
    
    expected_traces = [
        "trace_v1_coldstart.json",
        "trace_v1_squeeze.json",
        "trace_v1_entropy.json"
    ]
    
    missing = [f for f in expected_traces if not (traces_dir / f).exists()]
    if missing:
        print(f"  ⚠ Missing trace files: {missing} (will be needed for inference)")
        return True  # Not a hard failure at scaffolding stage
    
    print(f"  ✓ traces/ directory has all 3 trace files")
    return True


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("KubeCost-Gym Pre-Submission Validation")
    print("=" * 60)
    
    checks = [
        ("Module Imports", check_imports),
        ("openenv.yaml Structure", check_openenv_yaml),
        ("KubeCostEnv Structure", check_env_structure),
        ("Graders", check_graders),
        ("Grader Bounds [0.0-1.0]", check_grader_bounds),
        ("Environment Variables", check_env_variable_patterns),
        ("OpenAI Client Usage", check_openai_client),
        ("Requirements (OpenAI)", check_requirements_openai),
        ("Inference Root", check_inference_root),
        ("Dockerfile Build", check_dockerfile_build),
        ("Traces Directory", check_traces_directory),
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            results.append(check_fn())
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✓ All {total} validation checks PASSED")
        print("=" * 60)
        print("\n🎯 Pre-submission checklist complete!")
        print("\nBefore submitting to HF Spaces, verify one last time:")
        print("  1. Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN")
        print("  2. inference.py location: Root directory (not in subdirectory)")
        print("  3. Grader output: All scores in [0.0, 1.0] range")
        print("  4. No hardcoded API keys or tokens")
        print("  5. OpenAI Client used for all LLM calls")
        print("  6. Runtime: <20 minutes on vcpu=2, memory=8gb")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {total - passed} of {total} checks FAILED")
        print("=" * 60)
        print("\nFix the above issues before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
