# 🛠️ REMEDIATION PLAN: Hardening Against Guidelines

**Target:** `k8s-cost-optimizer` (specifically `inference.py`)  
**Objective:** Resolve all critical violations identified in `REVIEW.md` to ensure 100% compliance with `GUIDELINES.md` for OpenEnv telemetry.

---

## 🔧 FIX 1: Correct `[END]` Output Formatting

**File to modify:** `inference.py`

**Issue:** The `log_end` helper function prints an illegal `score=<score>` metric not recognized by the telemetry parser.

**Execution:**
1. Locate the `log_end` function around **Line 185** in `inference.py`.
2. Update the `[END]` logger print to strictly match the requested signature, removing the injected `score={score:.3f}` key-value pair.

**Concrete Diff:**
```diff
-def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
-    # Format: [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
-    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
-    success_val = str(success).lower()
-    print(
-        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
-        flush=True,
-    )

+def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
+    # Format: [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
+    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
+    success_val = str(success).lower()
+    # score removed from telemetry to match OpenEnv strict parsing rules
+    print(
+        f"[END] success={success_val} steps={steps} rewards={rewards_str}",
+        flush=True,
+    )
```

---

## 🔧 FIX 2: Environment Teardown (`env.close()`)

**File to modify:** `inference.py`

**Issue:** The environment is never properly closed prior to finalizing the episode logs. Since exceptions could occur during rollout, `env.close()` must be executed regardless of the terminal state. 

**Execution:**
1. Locate the `run_task` method of `CostOptimizerAgent` (starting around **Line 327**).
2. Instantiate `env = None` before the `try` block.
3. Guarantee that `env.close()` fires safely inside a `finally` block before logging the final `[END]` trace.

**Concrete Diff:**
```diff
    def run_task(self, task: Dict[str, Any]) -> float:
        task_name = task["name"]
        description = task["description"]
        grader = task["grader"]
        trace_path = task["trace"]

        log_start(task_name, self.model_name)

        total_steps = 0
        score = 0.0
        rewards = []
        success = False
+       env = None

        try:
            env = KubeCostEnv(trace_path)
            obs = env.reset()

            for step_num in range(1, MAX_STEPS_PER_TASK + 1):
                action = self.decide(obs, description)
                obs, reward, done, _info = env.step(action)
                total_steps = step_num
                rewards.append(float(reward))

                log_step(step_num, action.action_type.value, reward, done)

                if done:
                    break

            score = grader.grade(env.trajectory)
            score = max(0.1, min(0.9, score))
            success = score >= 0.1  # Standard success threshold

        except Exception as exc:
            print(
                f"[ERROR] Task '{task_name}' failed: {exc}", file=sys.stderr, flush=True
            )
            score = 0.1
            success = False
            
+       finally:
+           if env is not None:
+               env.close()

        log_end(success, total_steps, score, rewards)
        return score
```

---

## 🎯 Verification Checks (Post Run)

Once applied, test the inference script pipeline locally ensuring no errors are raised and validating that OpenEnv will accept the submission format:

1. **Verify Schema**:
   Run `python inference.py > output.txt 2> error.log` and verify that the `output.txt` only contains lines starting precisely with `[START]`, `[STEP]`, and `[END]`.

2. **Verify Pattern**: 
   Ensure `output.txt` matches:
   `[END] success=true steps=25 rewards=...` without the `score=` injection anywhere.

3. **Verify Lifecycle Integrity**: 
   Ensure no socket locks or orphaned environments persist in memory, inherently validated if `env.close()` fires successfully at runtime.
