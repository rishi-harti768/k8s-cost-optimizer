# KubeCost-Gym — Concrete Remediation Plan

**Linked Review:** `REVIEW.md`
**Date:** 2026-04-11
**Goal:** Harden the codebase against every finding. No TODOs — every fix is a drop-in patch.

Issue ordering follows severity: CRITICAL → HIGH → MEDIUM → LOW.

---

## BUG-01 · Terminal `step()` returns stale obs + zero reward

**File:** `server/k8s_cost_optimizer_environment.py` · Lines 402–406

**Problem:** When `_step >= total_steps`, the function exits early with the previous observation, a hardcoded `0.0` reward, an empty info dict, and skips the trajectory log.

**Fix:** Remove the early-exit shortcut. Let the normal flow complete for the last step, then set `done = True`. The `done` detection on Line 421 already handles this correctly.

```diff
-        self._step += 1
-
-        if self._step >= self.total_steps:
-            done = True
-            return self._current_obs, 0.0, done, {}
-
-        if self._current_obs is not None:
+        if self._step >= self.total_steps - 1:
+            # Already at the last valid step — episode is over, reject further calls
+            logger.warning("step() called after episode end; returning terminal state")
+            return self._current_obs, 0.0, True, {"step": self._step, "terminal": True}
+
+        self._step += 1
+
+        if self._current_obs is not None:
             self._prev_steal_pct = self._current_obs.cpu_steal_pct
```

Additionally add a post-episode guard at the top of `step()`:

```python
def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
    # Guard: reject calls after episode has ended
    if self._step >= self.total_steps - 1 and self._current_obs is not None:
        logger.warning("step() called on a finished episode. Call reset() first.")
        return self._current_obs, 0.0, True, {"step": self._step, "terminal": True}
    ...
```

**Expected outcome:** The final step's observation is built from trace data, reward is computed normally, the trajectory log captures it, and `done=True` is returned with a populated info dict.

---

## BUG-02 · Dead no-op in `_build_observation()` steal logic

**File:** `server/k8s_cost_optimizer_environment.py` · Lines 580–584

**Problem:** `max(x * 0.5, x)` always returns `x`. The intent was to halve steal when CPU is not under pressure.

```diff
-        else:
-            raw_steal_pct = max(
-                trace_obs.base_steal_pct * 0.5, trace_obs.base_steal_pct
-            )
+        else:
+            # CPU is not overloaded: reduce noisy-neighbour steal proportionally
+            raw_steal_pct = trace_obs.base_steal_pct * 0.5
```

**Expected outcome:** Steal is correctly reduced in the non-high-CPU branch. Task 2 (EfficientSqueeze) and the proactive bonus signal are now meaningful.

---

## BUG-06 · `check_imports()` is a hollow stub

**File:** `validate_local.py` · Lines 63–78

**Problem:** The function logs `[PASS]` without importing anything.

```diff
 def check_imports() -> bool:
     try:
-        logger.info("  [PASS] All modules import successfully")
-        return True
-    except ImportValidationError as e:
-        logger.error(f"  [FAIL] Import failed: {e}")
-        return False
-    except Exception as e:
-        logger.error(f"  [FAIL] Import failed: {e}")
-        return False
+        import models  # noqa: F401
+        import graders  # noqa: F401
+        from server.k8s_cost_optimizer_environment import K8sCostOptimizerEnvironment  # noqa: F401
+        logger.info("  [PASS] All modules import successfully")
+        return True
+    except ImportError as e:
+        logger.error(f"  [FAIL] Import failed: {e}")
+        return False
+    except Exception as e:
+        logger.error(f"  [FAIL] Unexpected import error: {e}")
+        return False
```

**Expected outcome:** If any module has a syntax error or missing dependency, `check_imports()` now surfaces the failure instead of lying about success.

---

## BUG-07 / TEST-01 · `test_env_contract.py` asserts wrong grader return values

**File:** `tests/test_env_contract.py` · Lines 51–56

**Problem:** Asserts `== 0.0` but all graders return `0.1` for empty trajectories.

```diff
 def test_graders_clamp_scores():
-    """Graders should clamp scores to [0.0, 1.0]."""
+    """Graders should return 0.1 for empty trajectories (spec §3 bounds [0.1, 0.9])."""
     empty = []
-    assert ColdStartGrader().grade(empty) == 0.0
-    assert EfficientSqueezeGrader().grade(empty) == 0.0
-    assert EntropyStormGrader().grade(empty) == 0.0
+    assert ColdStartGrader().grade(empty) == 0.1
+    assert EfficientSqueezeGrader().grade(empty) == 0.1
+    assert EntropyStormGrader().grade(empty) == 0.1
```

**Expected outcome:** Test passes and correctly documents the `[0.1, 0.9]` score contract.

---

## RTE-06 · `openenv.yaml` name mismatch with `validate_local.py`

Two independent fixes required:

### Fix A — Align the validator to the actual yaml name

**File:** `validate_local.py` · Line 101

```diff
-        if spec["name"] not in ["kubecost-gym", "kubecost_gym"]:
+        if spec["name"] not in ["kubecost-gym", "kubecost_gym", "k8s_cost_optimizer"]:
             raise ConfigValidationError(f"Invalid name: {spec['name']}")
```

### Fix B — Add the validator-required fields to `openenv.yaml`

**File:** `openenv.yaml`

```diff
 spec_version: 1
 name: k8s_cost_optimizer
+version: "0.1.0"
+description: "RL environment for proactive Kubernetes cost optimization."
 type: space
 runtime: fastapi
 app: server.app:app
 port: 7860
+tasks:
+  - name: cold_start
+    difficulty: easy
+    description: "Scale cluster from 0 to 5 replicas without SLA breach."
+  - name: efficient_squeeze
+    difficulty: medium
+    description: "Keep cpu_steal_pct < 20% across 24-hour sinusoidal load cycle."
+  - name: entropy_storm
+    difficulty: hard
+    description: "Issue REBALANCE_NODE before cpu_steal_pct exceeds 20%."
```

**Expected outcome:** `check_openenv_yaml()` passes without errors.

---

## BUG-08 · `verify_remote.py` sends an invalid action payload

**File:** `verify_remote.py` · Line 66

**Problem:** `"SCALE_REPLICAS"` does not exist in `ActionType`; `"value"` is not a model field.

```diff
-    action_payload = {"action": {"action_type": "SCALE_REPLICAS", "value": 5}}
+    action_payload = {"action": {"action_type": "SCALE_REPLICAS(+5)"}}
```

**Expected outcome:** The remote step test sends a valid `ActionType` and the server processes it successfully.

---

## BUG-04 · `COST_PENALTY_CAP` is dead configuration

**File:** `server/k8s_cost_optimizer_environment.py`

Two options — choose one based on design intent:

### Option A — Apply the cap (the constant implies it should be applied)

```diff
     # Cost penalty
-    cost_penalty = _CONFIG.COST_PENALTY_RATE * cost_fraction
+    cost_penalty = min(
+        _CONFIG.COST_PENALTY_RATE * cost_fraction,
+        _CONFIG.COST_PENALTY_CAP,
+    )
```

### Option B — Remove the dead constant (if uncapped is intentional)

```diff
 class _EnvironmentConfig:
     UPTIME_REWARD: float = 10.0
     COST_PENALTY_RATE: float = 5.0
-    COST_PENALTY_CAP: float = 5.0  # Remove this line entirely
     RAMP_PENALTY_RATE: float = 5.0
```

**Recommendation:** Apply Option A, which matches the name of the constant and gives agents a predictable worst-case cost signal.

**Expected outcome:** Either the cap is enforced consistently, or the constant is removed and the code matches the comment.

---

## LOGIC-01 / RTE-01 · `step()` can `IndexError` after episode ends

**File:** `server/k8s_cost_optimizer_environment.py`

The combined fix from BUG-01 already addresses the primary case. Add an additional explicit guard at Line 414:

```diff
+        if self._step >= self.total_steps:
+            raise EnvError(
+                f"step() called with _step={self._step} >= total_steps={self.total_steps}. "
+                "Call reset() to start a new episode."
+            )
         trace_step = self.steps_data[self._step]
```

**Expected outcome:** Any call to `step()` after `done=True` raises an explicit, descriptive `EnvError` instead of a raw `IndexError`.

---

## LOGIC-04 · Perfect proactive agent scores `0.1` (inverted incentive)

**File:** `graders.py` · Lines 268–271

**Problem:** If no violations ever occur, the grader returns `0.1` regardless of how many proactive `REBALANCE_NODE` actions the agent took.

```diff
         # Special case: zero violations
         if not violation_indices:
-            # No violations means there was no observed breach to credit.
-            return 0.1
+            # Check if the agent took proactive REBALANCE_NODE actions on rising steal.
+            # An agent that genuinely suppressed all spikes should be credited.
+            proactive_count = sum(
+                1
+                for i, step in enumerate(trajectory)
+                if step.action == ActionType.REBALANCE_NODE
+                and (
+                    i == 0
+                    or trajectory[i].observation.cpu_steal_pct
+                    > trajectory[i - 1].observation.cpu_steal_pct
+                )
+            )
+            if proactive_count > 0:
+                # Reward proactive suppression: scale by fraction of steps with rising steal
+                rising_steal_steps = sum(
+                    1
+                    for i in range(1, len(trajectory))
+                    if trajectory[i].observation.cpu_steal_pct
+                    > trajectory[i - 1].observation.cpu_steal_pct
+                )
+                if rising_steal_steps > 0:
+                    success_rate = min(1.0, proactive_count / rising_steal_steps)
+                    return max(0.1, min(0.9, success_rate))
+            # No violations AND no proactive actions: passive agent
+            return 0.1
```

**Expected outcome:** An agent that proactively rebalances on rising steal signals — preventing all breaches — receives a score above `0.1`. A truly passive agent with no violations still scores `0.1`.

---

## BUG-05 · `_is_rising_steal` closure redefined inside a loop

**File:** `graders.py` · Lines 284–296

```diff
+    def _is_rising_steal(step_index: int) -> bool:
+        """Return True if steal at step_index is higher than the preceding step."""
+        if step_index == 0:
+            return trajectory[step_index].observation.cpu_steal_pct > 0.0
+        return (
+            trajectory[step_index].observation.cpu_steal_pct
+            > trajectory[step_index - 1].observation.cpu_steal_pct
+        )
+
     for violation_idx in violation_indices:
         window_start = max(0, violation_idx - _CONFIG.LOOKBACK_WINDOW)
         window_end = violation_idx

-        def _is_rising_steal(step_index: int) -> bool:
-            if step_index == 0:
-                return trajectory[step_index].observation.cpu_steal_pct > 0.0
-            return (
-                trajectory[step_index].observation.cpu_steal_pct
-                > trajectory[step_index - 1].observation.cpu_steal_pct
-            )
-
         rebalanced_proactively = any(
```

**Expected outcome:** The helper is defined once, allocated once, is clearly visible at grader scope, and is safe from any future loop-variable capture bugs.

---

## TEST-02 · Wrong enum vs string comparison in `test_rebalance_node_action`

**File:** `tests/test_env_contract.py` · Line 139

**Problem:** `TrajectoryStep` uses `use_enum_values=True`, so `.action` is the string `"REBALANCE_NODE"`, not the enum member.

```diff
-    assert trajectory[-1].action == ActionType.REBALANCE_NODE
+    assert trajectory[-1].action == ActionType.REBALANCE_NODE.value
```

**Expected outcome:** Comparison is string-to-string; the assertion passes.

---

## HC-02 · `BUDGET` duplicated between `graders.py` and `env.py`

**Problem:** Two separate `100.0` constants can silently diverge.

**Fix:** Define the budget once in `models.py` (or a new `constants.py`) and import it everywhere.

### Step 1 — Add to `models.py` (bottom, before `__all__` or after enums)

```python
# ===== SHARED CONSTANTS =====

HOURLY_BUDGET: float = 100.0
"""Shared budget constant for reward and grader normalization."""
```

### Step 2 — Update `graders.py`

```diff
-from models import TrajectoryStep, ActionType
+from models import TrajectoryStep, ActionType, HOURLY_BUDGET

 class _GraderConfig:
-    BUDGET: float = 100.0
+    BUDGET: float = HOURLY_BUDGET
```

### Step 3 — Update `server/k8s_cost_optimizer_environment.py`

```diff
-from models import (Observation, Action, EnvState, ...)
+from models import (Observation, Action, EnvState, ..., HOURLY_BUDGET)

 class _EnvironmentConfig:
-    HOURLY_BUDGET: float = 100.0
+    HOURLY_BUDGET: float = HOURLY_BUDGET
```

**Expected outcome:** A single source of truth for the $100 budget. Changing it in one place propagates everywhere automatically.

---

## DESIGN-02 · `graders.py` bare import fails when installed as package

**File:** `graders.py` · Line 19

```diff
-from models import TrajectoryStep, ActionType
+try:
+    from k8s_cost_optimizer.models import TrajectoryStep, ActionType
+except ImportError:
+    from models import TrajectoryStep, ActionType
```

**Expected outcome:** `graders.py` imports correctly both when run from the repo root (local dev/tests) and when installed as the `k8s_cost_optimizer` package on HF Spaces or Docker.

---

## DESIGN-04 · `validate_env()` logic is inconsistent with pipeline spec

**File:** `inference.py` · Lines 372–379

**Problem:** The submission pipeline requires defaults for `API_BASE_URL` and `MODEL_NAME`. `validate_env()` raises on their absence, which would incorrectly fail valid submissions.

```diff
 def validate_env() -> None:
-    missing = [
-        k for k in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(k)
-    ]
-    if missing:
-        raise EnvironmentValidationError(
-            f"Missing required environment variables: {', '.join(missing)}"
-        )
+    # Per submission pipeline spec: API_BASE_URL and MODEL_NAME have mandatory
+    # defaults and need not be set explicitly. Only HF_TOKEN is strictly required.
+    if not os.environ.get("HF_TOKEN"):
+        raise EnvironmentValidationError(
+            "HF_TOKEN is required but not set. "
+            "Set it with: export HF_TOKEN=<your-api-key>"
+        )
```

**Expected outcome:** `validate_env()` is now consistent with the pipeline rules and can be safely called from `main()` or any local validation script.

### Bonus: wire `validate_env()` into `main()`

Replace the manual HF_TOKEN check in `main()`:

```diff
 def main() -> None:
-    if not os.environ.get("HF_TOKEN"):
-        print("[ERROR] HF_TOKEN environment variable is required", file=sys.stderr)
-        print("[ERROR] Set it with: set HF_TOKEN=your-api-key", file=sys.stderr)
-        print("[ERROR] Or for testing: set HF_TOKEN=dummy-token", file=sys.stderr)
-        sys.exit(1)
+    try:
+        validate_env()
+    except EnvironmentValidationError as exc:
+        print(f"[ERROR] {exc}", file=sys.stderr)
+        sys.exit(1)
```

---

## DESIGN-03 · `steal_suppression_steps` missing from `state()` snapshot

**File:** `server/k8s_cost_optimizer_environment.py`

### Step 1 — Add field to `EnvState` in `models.py`

```diff
 class EnvState(BaseModel):
     step: int = Field(ge=0, description="Current step counter [0-∞]")
     replicas: int = Field(ge=0, description="Active replica count [0-∞]")
     node_size: NodeSizeClass = Field(description="Current node tier {S|M|L}")
     prev_steal_pct: float = Field(ge=0, le=1, description="Previous-step steal % [0-1]")
+    steal_suppression_steps: int = Field(
+        ge=0,
+        default=0,
+        description="Remaining steps of REBALANCE steal suppression [0-3]",
+    )
```

### Step 2 — Include it in `state()`

```diff
     def state(self) -> EnvState:
         return EnvState(
             step=self._step,
             replicas=self._replicas,
             node_size=self._node_size,
             prev_steal_pct=self._prev_steal_pct,
+            steal_suppression_steps=self.steal_suppression_steps,
         )
```

**Expected outcome:** Callers of `state()` can observe whether steal suppression is currently active, enabling better debugging and testing.

---

## DESIGN-01 · `reward` and `done` embedded in `Observation`

**File:** `models.py` · Lines 99–106

**Problem:** Observations should represent pure environment state. `reward` and `done` are step-result fields.

**Fix:** Keep the fields for now but mark them clearly as compatibility-only and ensure they are excluded from agent-facing serialization where possible.

```diff
     node_bin_density: conlist(...) = Field(...)
+
+    # ------------------------------------------------------------------
+    # OpenEnv harness compatibility fields — NOT part of the observable state.
+    # These exist solely because openenv-core's reset() response serializer
+    # expects them. Do NOT use these fields in reward computation or grading.
+    # ------------------------------------------------------------------
     reward: float = Field(
         default=0.0,
-        description="Reward signal for the step (OpenEnv reset response compatibility)",
+        description="[COMPAT ONLY] Harness reward slot — not observable state.",
+        exclude=True,   # exclude from agent obs dumps by default
     )
     done: bool = Field(
         default=False,
-        description="Episode termination flag (OpenEnv reset response compatibility)",
+        description="[COMPAT ONLY] Harness done slot — not observable state.",
+        exclude=True,
     )
```

> **Note:** If `openenv-core` serializes the full model and requires these fields to be present in the JSON output, remove `exclude=True` and add a note explaining they are always `0.0` / `False` at reset time.

**Expected outcome:** Architectural intent is explicit; these fields are invisible in `obs.model_dump()` unless explicitly requested.

---

## RTE-04 · `client.py` — `reward` can be `None`

**File:** `client.py` · Line 31

```diff
         return StepResult(
             observation=observation,
-            reward=payload.get("reward"),
+            reward=float(payload.get("reward", 0.0)),
             done=payload.get("done", False),
         )
```

**Expected outcome:** `reward` is always a `float`, never `None`, preventing downstream type errors.

---

## RTE-05 · `_apply_action()` silently ignores unknown `ActionType`

**File:** `server/k8s_cost_optimizer_environment.py` · end of `_apply_action()`

```diff
         # ---- MAINTAIN: explicit no-op ----
         elif action_type == ActionType.MAINTAIN:
             pass
+
+        else:
+            # Should be unreachable if Pydantic validation works correctly.
+            # Log loudly if a new ActionType is added without updating this method.
+            logger.error(
+                f"_apply_action: unhandled ActionType '{action_type}'. "
+                "Treating as MAINTAIN. Update _apply_action() to handle new actions."
+            )
```

**Expected outcome:** Adding a new `ActionType` to the enum without updating `_apply_action` now logs a loud error rather than silently degrading.

---

## HC-05 · Reward weights not externally configurable

**File:** `server/k8s_cost_optimizer_environment.py` · `_EnvironmentConfig`

**Fix:** Source weights from environment variables with the existing hardcoded values as defaults.

```diff
 class _EnvironmentConfig:
-    UPTIME_REWARD: float = 10.0
-    COST_PENALTY_RATE: float = 5.0
-    RAMP_PENALTY_RATE: float = 5.0
-    SLA_BREACH_PENALTY: float = 20.0
-    PROACTIVE_BONUS: float = 0.5
+    import os as _os
+    UPTIME_REWARD: float = float(_os.getenv("ENV_UPTIME_REWARD", "10.0"))
+    COST_PENALTY_RATE: float = float(_os.getenv("ENV_COST_PENALTY_RATE", "5.0"))
+    RAMP_PENALTY_RATE: float = float(_os.getenv("ENV_RAMP_PENALTY_RATE", "5.0"))
+    SLA_BREACH_PENALTY: float = float(_os.getenv("ENV_SLA_BREACH_PENALTY", "20.0"))
+    PROACTIVE_BONUS: float = float(_os.getenv("ENV_PROACTIVE_BONUS", "0.5"))
```

> **Note:** Import `os` at the top of the file normally; the inline `import os as _os` is shown for clarity of the class-scope trick. Use a class-level method or module-level import in practice.

**Expected outcome:** Reward weights can be varied for experiments via environment variables without code changes. Defaults are unchanged, so existing tests pass unmodified.

---

## HC-01 & LOGIC-05 · `HOURLY_BUDGET` and node pricing not configurable

**File:** `server/k8s_cost_optimizer_environment.py`

These are covered partially by HC-02 (shared budget constant) and are lower priority since:
- The budget is shared via the `models.py` constant fix above.
- Node base costs are architectural constants that reflect the spec's pricing model.

**Minimum fix for node costs** — annotate as intentional:

```diff
     def _compute_current_cost(self) -> float:
+        # Pricing model per spec §4: S=$10/hr, M=$20/hr, L=$40/hr + $1/replica.
+        # These are intentionally fixed to keep the reward signal deterministic.
         base_costs = {
             NodeSizeClass.SMALL: 10.0,
             NodeSizeClass.MEDIUM: 20.0,
             NodeSizeClass.LARGE: 40.0,
         }
         return base_costs.get(self._node_size, 10.0) + float(self._replicas)
```

---

## LOGIC-06 · `buffer_depth` grows unboundedly in traces

**File:** `generate_traces.py` · Line 77

```diff
-            "buffer_depth": 80 + i * 3,
+            "buffer_depth": min(300, 80 + i * 3),  # cap at 300 to prevent observation explosion
```

And add a corresponding upper-bound validation in `TraceObservation`:

```diff
-    buffer_depth: int = Field(ge=0, description="Baseline queue depth for the raw workload")
+    buffer_depth: int = Field(ge=0, le=500, description="Baseline queue depth [0-500]")
```

**Expected outcome:** `buffer_depth` is bounded, preventing runaway values from polluting observations at longer trace lengths.

---

## LOGIC-03 · `SUPPORTS_CONCURRENT_SESSIONS` vs `max_concurrent_envs` conflict

**File:** `server/app.py` · Line 21
**File:** `server/k8s_cost_optimizer_environment.py` · Line 305

The simplest fix is to ensure both agree:

```diff
-    SUPPORTS_CONCURRENT_SESSIONS: bool = False
+    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # openenv-core uses max_concurrent_envs=1 as the limiter
```

Or alternatively, if truly single-session:

```diff
 app = create_app(
     K8sCostOptimizerEnvironment,
     Action,
     Observation,
     env_name="k8s_cost_optimizer",
-    max_concurrent_envs=1,
+    max_concurrent_envs=0,  # 0 = unlimited (serialized by SUPPORTS_CONCURRENT_SESSIONS=False)
 )
```

**Recommendation:** Verify `openenv-core` documentation for exact semantics of `max_concurrent_envs=0` vs `=1`. Set `SUPPORTS_CONCURRENT_SESSIONS=True` and keep `max_concurrent_envs=1` to express "at most 1 concurrent session" clearly.

---

## LOGIC-02 · Reward ordering subtlety — document the causal chain

**File:** `server/k8s_cost_optimizer_environment.py` · Lines 408–420

No code change needed — document the ordering explicitly to prevent future regressions:

```diff
+        # Causal chain for proactive bonus (order is load-bearing):
+        #   1. Capture steal from CURRENT obs as "previous" for this step's comparison.
+        #   2. Apply action — REBALANCE sets steal_suppression_steps=3.
+        #   3. Build NEW obs — suppression reduces steal in _build_observation.
+        #   4. Reward uses NEW obs steal vs captured "previous" steal → bonus fires at t+1.
         if self._current_obs is not None:
             self._prev_steal_pct = self._current_obs.cpu_steal_pct
```

---

## TEST-03 · Implicit trace content dependency in scale test

**File:** `tests/test_env_contract.py` · Lines 116–125

```diff
 def test_scale_actions_modify_replica_count():
     """Scale actions should modify internal replica count."""
     env = KubeCostEnv("traces/trace_v1_coldstart.json")
-    env.reset()
+    obs = env.reset()
+    initial_replicas = env.state().replicas  # Don't assume 0 — read from trace

     # Scale up
     obs1, _, _, _ = env.step(Action(action_type=ActionType.SCALE_UP_5))
     state1 = env.state()
-    assert state1.replicas == 5
+    assert state1.replicas == initial_replicas + 5

     obs2, _, _, _ = env.step(Action(action_type=ActionType.SCALE_UP_5))
     state2 = env.state()
-    assert state2.replicas == 10
+    assert state2.replicas == initial_replicas + 10

     obs3, _, _, _ = env.step(Action(action_type=ActionType.SCALE_DOWN_1))
     state3 = env.state()
-    assert state3.replicas == 9
+    assert state3.replicas == initial_replicas + 9
```

**Expected outcome:** Test is trace-agnostic and will not break if traces are regenerated with different initial replica counts.

---

## TEST-04 · CWD-relative trace paths in fixture

**File:** `tests/test_inference_integration.py` · Lines 20–28

```diff
+import os
+from pathlib import Path

+REPO_ROOT = Path(__file__).parent.parent

 @pytest.fixture
 def traces_available():
     """Check if trace files exist."""
     traces = [
-        "traces/trace_v1_coldstart.json",
-        "traces/trace_v1_squeeze.json",
-        "traces/trace_v1_entropy.json",
+        REPO_ROOT / "traces/trace_v1_coldstart.json",
+        REPO_ROOT / "traces/trace_v1_squeeze.json",
+        REPO_ROOT / "traces/trace_v1_entropy.json",
     ]
-    return all(os.path.exists(t) for t in traces)
+    return all(t.exists() for t in traces)
```

Also update `KubeCostEnv` instantiation calls in the same file to use absolute paths:

```diff
-        env = KubeCostEnv("traces/trace_v1_coldstart.json")
+        env = KubeCostEnv(str(REPO_ROOT / "traces/trace_v1_coldstart.json"))
```

**Expected outcome:** Tests pass regardless of which directory `pytest` is invoked from.

---

## HC-06 / DESIGN-04 · `validate_env()` scope (confirmed pipeline-compliant, no fix needed)

As clarified: defaults for `API_BASE_URL` and `MODEL_NAME` are **required by the submission pipeline**. The fix in DESIGN-04 above (scope `validate_env()` to `HF_TOKEN` only) is the only change needed here. No further action.

---

## RTE-02 · Silent `.env` parse failures

**File:** `inference.py` · Lines 117–119

Already handles `maxsplit=1` correctly. Add only a final warning if variables remain unresolved:

```diff
     load_env()
+    # Warn if any expected variable is still unset after .env loading
+    for var in ("API_BASE_URL", "MODEL_NAME"):
+        if not os.environ.get(var):
+            print(
+                f"[INFO] {var} not set; using built-in default. "
+                "Set explicitly to override.",
+                file=sys.stderr,
+            )
```

---

## RTE-03 · `hf_logs.py` — `HF_USER_TOKEN` type annotation

**File:** `hf_logs.py` · Line 12

```diff
-HF_USER_TOKEN = os.environ.get("HF_USER_TOKEN")
+HF_USER_TOKEN: str | None = os.environ.get("HF_USER_TOKEN")
```

Minor — adds type clarity for static analysis tools.

---

## BUG-03 · No guard against calling `step()` before `reset()`

**File:** `server/k8s_cost_optimizer_environment.py` · top of `step()`

```diff
     def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
+        if self._current_obs is None:
+            raise EnvError(
+                "step() called before reset(). Call env.reset() to initialize the episode."
+            )
         # Guard: reject calls after episode has ended (already added in BUG-01 fix)
```

**Expected outcome:** A clean, descriptive error fires instead of a `TypeError` or `AttributeError` if `step()` is called on a fresh environment.

---

## DESIGN-05 · `reset()` defensive guard for empty trace

**File:** `server/k8s_cost_optimizer_environment.py` · top of `reset()`

```diff
     def reset(self, **kwargs) -> Observation:
+        if not self.steps_data:
+            raise TraceLoadError(
+                f"Trace '{self.trace_path}' contains no steps. Cannot reset."
+            )
         self._step = 0
```

Low risk since Pydantic enforces `min_length=1` on load, but defence-in-depth is worth the one line.

---

## HC-04 · Document capacity multiplier asymmetry

**File:** `server/k8s_cost_optimizer_environment.py` · Lines 550–556

No code change — add a comment explaining the intentional asymmetry:

```diff
     def _get_node_capacity_multiplier(self) -> float:
-        """Return capacity multiplier for the current node size."""
+        """Return capacity multiplier for the current node size.
+
+        Intentional asymmetry (per spec §4):
+          SMALL  → 1×  (baseline)
+          MEDIUM → 2×  (double core count)
+          LARGE  → 4×  (quad core + memory headroom)
+        This reflects a realistic cloud node tier ratio (e.g., 2vCPU→4vCPU→8vCPU).
+        """
         if self._node_size == NodeSizeClass.SMALL:
```

---

## Summary — Fix Priority Order

| Priority | ID | File | Fix Type |
|---|---|---|---|
| 1 | BUG-01 | env.py | Remove early-exit; fix terminal step |
| 2 | BUG-02 | env.py | Fix dead `max()` → use `* 0.5` directly |
| 3 | BUG-06 | validate_local.py | Implement `check_imports()` body |
| 4 | BUG-07 / TEST-01 | tests/test_env_contract.py | Fix `== 0.0` → `== 0.1` |
| 5 | RTE-06 | validate_local.py + openenv.yaml | Align names; add required yaml fields |
| 6 | BUG-08 | verify_remote.py | Fix invalid action payload |
| 7 | BUG-04 | env.py | Apply or remove `COST_PENALTY_CAP` |
| 8 | LOGIC-01 / RTE-01 | env.py | Add post-episode IndexError guard |
| 9 | LOGIC-04 | graders.py | Fix inverted incentive for perfect proactive agent |
| 10 | BUG-05 | graders.py | Move `_is_rising_steal` out of loop |
| 11 | TEST-02 | tests/test_env_contract.py | Fix enum vs string comparison |
| 12 | HC-02 | models.py + graders.py + env.py | Single shared `HOURLY_BUDGET` constant |
| 13 | DESIGN-02 | graders.py | Add `try/except ImportError` fallback |
| 14 | DESIGN-04 | inference.py | Scope `validate_env()` to HF_TOKEN only + wire into `main()` |
| 15 | DESIGN-03 | models.py + env.py | Add `steal_suppression_steps` to `EnvState` |
| 16 | DESIGN-01 | models.py | Mark `reward`/`done` as compat-only + `exclude=True` |
| 17 | RTE-04 | client.py | `float(payload.get("reward", 0.0))` |
| 18 | RTE-05 | env.py | Add `else: logger.error(...)` in `_apply_action` |
| 19 | HC-05 | env.py | Source reward weights from env vars |
| 20 | LOGIC-06 | generate_traces.py + models.py | Cap `buffer_depth` |
| 21 | LOGIC-03 | env.py / app.py | Align concurrent session config |
| 22 | LOGIC-02 | env.py | Add causal chain comment |
| 23 | TEST-03 | tests/test_env_contract.py | Trace-agnostic replica assertions |
| 24 | TEST-04 | tests/test_inference_integration.py | Absolute path fixture |
| 25 | HC-01 / LOGIC-05 | env.py | Annotate pricing as spec-intentional |
| 26 | BUG-03 | env.py | Guard `step()` before `reset()` |
| 27 | DESIGN-05 | env.py | Guard `reset()` on empty trace |
| 28 | HC-04 | env.py | Document multiplier asymmetry |
| 29 | RTE-02 | inference.py | Warn on unset optional vars post-load |
| 30 | RTE-03 | hf_logs.py | Type-annotate `HF_USER_TOKEN` |

---

*End of Remediation Plan*
