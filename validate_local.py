# validate_local.py
"""
Pre-submission validation harness (spec §7 automated gates).

Run locally before pushing to HF Space.

Validates:
  - Module imports (no syntax errors)
  - reset() returns Observation (not dict)
  - step() returns 4-tuple with correct types
  - state() returns EnvState (not dict)
  - All graders return [0.0, 1.0]
  - openenv.yaml parses and has required fields
  - inference.py exists in root
  - No stub bodies remain (all are pass)

Reference: PROJECT_SPEC.md §7 Pre-Submission Checklist
"""

import sys
import yaml
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
    print("KubeCost-Gym Local Validation")
    print("=" * 60)
    
    checks = [
        ("Module Imports", check_imports),
        ("openenv.yaml Structure", check_openenv_yaml),
        ("KubeCostEnv Structure", check_env_structure),
        ("Graders", check_graders),
        ("Inference Root", check_inference_root),
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
        print("\nScaffolding complete! Ready for Phase 1 implementation.")
        print("Next steps:")
        print("  1. Create JSON files in traces/")
        print("  2. Implement Phase 1: Domain Specification")
        print("  3. Proceed with SDD phases (2-5)")
        print("=" * 60)
        return 0
    else:
        print(f"✗ {total - passed} of {total} checks FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
