# Spec-Driven Development for OpenEnv — Project Specification

**Project:** KubeCost-Gym v3.1 — Kubernetes Cost Optimization Environment  
**Deadline:** 8 Apr 2026, 11:59 PM IST  
**Target Score:** ≥27/30  
**Framework:** OpenEnv · HuggingFace Spaces · Python 3.10+

---

## 1. Overview

This project applies **Spec-Driven Development (SDD)** — a methodology where precise specifications are written before implementation. The core insight: bugs are rarely caused by incorrect logic inside functions, but by mismatched expectations at component boundaries.

The submission contains:

- A production-grade Kubernetes cluster autoscaling simulator
- Typed Pydantic models for all data structures
- Mathematical specifications for reward and grading functions
- Dockerized inference pipeline using LLM agents
- Three progressive difficulty tasks testing different decision-making strategies

---

## 2. The OpenEnv Spec — Mandatory Interface

All submitted environments must implement this interface:

| Method/File    | Signature                             | Validator Checks                         | Common Failure                               |
| -------------- | ------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| `reset()`      | `() → Observation`                    | Returns valid Pydantic model             | Returns dict instead                         |
| `step(action)` | `(Action) → (Obs, float, bool, dict)` | 4-tuple; reward is float; done is bool   | Stub returns None                            |
| `state()`      | `() → typed model or dict`            | Non-null, serializable                   | Returns untyped dict                         |
| `openenv.yaml` | name, version, description, tasks[]   | YAML parses; required keys present       | Missing task difficulty                      |
| `graders`      | `(trajectory) → float`                | Returns [0.0, 1.0]                       | Score > 1.0                                  |
| `inference.py` | Root directory; uses env vars         | File exists, runs cleanly, <20 min       | In subdirectory                              |
| `README.md`    | HF Space frontmatter                  | hardware tag, sdk: docker, tags: openenv | hardware: cpu-upgrade (violates 2vCPU limit) |

### The openenv.yaml Structure

```yaml
name: your-env-name # kebab-case, unique
version: "3.0" # STRING, not bare float
description: "One sentence." # required
tasks:
  - name: task_one
    difficulty: easy # easy | medium | hard
    description: "What the agent must do."
  - name: task_two
    difficulty: medium
    description: "Specific success criteria."
  - name: task_three
    difficulty: hard
    description: "Proactive or multi-step challenge."
```

**Critical:** YAML version numbers like `3.0` parse as floats. Always quote: `version: "3.0"`.

---

## 3. The 5 SDD Phases

### Phase Overview

```
Domain Spec → Contract Spec → Reward Spec → Grader Spec → Infra Spec
     ↓              ↓               ↓              ↓            ↓
  Problem      Type Models      Formula      Normalized    Docker &
  Definition                    Design       Scoring       Deployment
```

### Phase 1: Domain Specification

**Goal:** Define the real-world problem and design the task ladder.

**Template:**

- **Real-World Problem:** What system does this simulate? Kubernetes autoscaling balances cost (fewer nodes) vs. reliability (enough capacity).
- **Observable Variables:** List every field an agent can perceive with units and ranges
  - Example: `cpu_usage_pct: 0–100%`, `p99_latency_ms: 0–∞ ms`, `cpu_steal_pct: 0–1`
- **Controllable Variables (Actions):** What levers can the agent pull?
  - Example: `SCALE_REPLICAS(±N)`, `REBALANCE_NODE`, `UPGRADE_NODE`
- **Good vs. Bad Outcomes:** Define in English before any formula
- **Task Ladder:**
  - **Easy:** Reactive solution works. Success is immediate and obvious (e.g., "Scale to 5 replicas in 3 steps")
  - **Medium:** Agent must balance competing objectives over time (e.g., "Maintain density AND minimize resource waste across 24h sine wave")
  - **Hard:** Agent must act BEFORE problems are visible using leading indicators (e.g., "Issue rebalance before steal exceeds 20%")
- **Determinism Guarantee:** How to ensure reproducibility? (e.g., "All dynamics from pre-recorded JSON traces, no RNG")

**Design Principle:** The hard task should be impossible for reactive agents but solvable for agents that learn latent structure.

### Phase 2: Contract Specification

**Goal:** Translate domain spec into typed Pydantic models.

**Key Rules:**

- Every observation field has both type annotation AND `Field()` constraint
- No bare strings — use `Enum` for finite value sets
- Fixed-length lists: `min_length = max_length`
- `state()` returns typed Pydantic model, NOT dict
- Never compare floats with `==` — use tolerance: `abs(x - target) < 1e-3`

**Example:**

```python
from pydantic import BaseModel, Field
from enum import Enum

class NodeSize(str, Enum):
    S = "S"
    M = "M"
    L = "L"

class Observation(BaseModel):
    cpu_usage_pct: float = Field(ge=0, le=100)        # Percentages: 0–100
    p99_latency_ms: float = Field(ge=0)               # Latency: non-negative
    http_error_rate: float = Field(ge=0, le=1)        # Ratios: 0–1
    active_replicas: int = Field(ge=0)                # Counts: non-negative
    node_size_class: NodeSize                          # Enum: constrained values
    node_bin_density: List[float] = Field(min_length=10, max_length=10)  # Fixed length

class ActionType(str, Enum):
    SCALE_REPLICAS_N5 = "SCALE_REPLICAS(-5)"
    MAINTAIN = "MAINTAIN"
    SCALE_REPLICAS_P5 = "SCALE_REPLICAS(+5)"
    # ... comprehensive enumeration

class EnvState(BaseModel):
    step: int
    replicas: int
    node_size: NodeSize
```

### Phase 3: Reward Specification

**Goal:** Design the learning signal with mathematical specification before implementation.

**Rules:**

- ✓ No sparse cliffs — add gradient ramps approaching hard thresholds
- ✓ Bounded magnitude — specify R_min and R_max per step
- ✓ Competing objectives — cost AND SLA should both contribute
- ✓ Proactive bonuses — reward early corrective action on leading indicators
- ✗ Never reward actions; reward outcomes

**KubeCost-Gym Reward Formula:**

```
R = (10.0 × Uptime) − (5.0 × Cost/Budget) − RampPenalty(p99) − SLABreach(p99) + ProactiveBonus

Where:
  Uptime = 1.0 if p99 < 300ms, else 0.0
  RampPenalty(p99) = (p99−200)/100 × 5.0 when p99 ∈ [200, 300ms)  ← gradient signal
  SLABreach = −20.0 when p99 ≥ 300ms
  ProactiveBonus = +0.5 when steal drops AND p99 < 300ms

  R_min per step: −20.0
  R_max per step: +10.5
```

**The Ramp Insight:** Binary reward at 300ms makes agents learn to avoid the entire warning zone. A linear ramp penalty for [200, 300ms) creates continuous gradient, enabling dense and informative learning signals.

### Phase 4: Grader Specification

**Goal:** Specify normalized scoring functions that always return 0.0–1.0.

**Critical Rules:**

1. **Normalization:** Score must be invariant to trajectory length
   - ✗ Wrong: `score = 1.0; score -= 0.05 per violation` (unbounded, varies by length)
   - ✓ Correct: `score = 1.0 - (violations / len(trajectory))`

2. **Float comparisons:** Use tolerance, never `==`
   - ✗ `error_rate == 0.0`
   - ✓ `error_rate < 0.001`

3. **Edge cases:**
   - Empty trajectory: return 0.0 explicitly
   - Final clamp: `max(0.0, min(1.0, raw_score))`

**Example:**

```python
class EfficientSqueezeGrader:
    def grade(self, trajectory: list) -> float:
        if not trajectory:
            return 0.0
        violations = sum(
            1 for step in trajectory
            if step["observation"].cpu_steal_pct >= 0.01
        )
        score = 1.0 - (violations / len(trajectory))
        return max(0.0, min(1.0, score))
```

### Phase 5: Infrastructure Specification

**Goal:** Containerize and deploy to HuggingFace Spaces.

**Dockerfile:**

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "inference.py"]
```

**README.md Frontmatter:**

```yaml
---
title: your-env-name
emoji: 🚀
colorFrom: red
colorTo: orange
sdk: docker
hardware: cpu-basic # ✓ CORRECT (2 vCPU / 8 GB RAM)
# hardware: cpu-upgrade          # ✗ WRONG (8 vCPU exceeds 2vCPU limit)
tags:
  - openenv
---
```

**inference.py Contract:**

- Located in project root (not subdirectory)
- Reads exactly: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` from environment
- Uses OpenAI client only
- Runs end-to-end in <20 minutes
- Must complete when executed: `if __name__ == "__main__"`

---

## 4. Environment Interface: KubeCost-Gym

### Observation Space

| Field                 | Type        | Range       | Purpose                                   |
| --------------------- | ----------- | ----------- | ----------------------------------------- |
| `cpu_usage_pct`       | float       | [0, 100]    | Cluster-wide CPU utilization              |
| `mem_usage_pct`       | float       | [0, 100]    | Cluster-wide memory utilization           |
| `p99_latency_ms`      | float       | [0, ∞)      | Tail latency; SLA threshold = 300ms       |
| `http_error_rate`     | float       | [0, 1]      | Request failure rate                      |
| `cpu_steal_pct`       | float       | [0, 1]      | Noisy-neighbor indicator                  |
| `active_replicas`     | int         | [0, ∞)      | Running pod count                         |
| `buffer_depth`        | int         | [0, ∞)      | Request queue depth                       |
| `node_size_class`     | Enum(S/M/L) | {S, M, L}   | Current node tier                         |
| `current_hourly_cost` | float       | [0, ∞)      | USD/hour spend                            |
| `node_bin_density`    | List[float] | [0, 1] × 10 | Per-node packing; fixed 10-element vector |

### Action Space

```
SCALE_REPLICAS(-5)    : aggressive scale-down
SCALE_REPLICAS(-1)    : gentle scale-down
MAINTAIN              : hold current state
SCALE_REPLICAS(+1)    : gentle scale-up
SCALE_REPLICAS(+5)    : moderate scale-up
SCALE_REPLICAS(+10)   : large scale-up
SCALE_REPLICAS(+20)   : emergency burst absorption (CRITICAL: hard task requires this)
UPGRADE_NODE          : vertical scale (irreversible for 1 step)
REBALANCE_NODE        : deterministic rebalance
```

### Environment State

Returns typed `EnvState` model:

- `step`: current step counter
- `replicas`: active replica count
- `node_size`: current NodeSize
- `prev_steal_pct`: previous-step steal percentage (for proactive bonus detection)

---

## 5. Critical SDD Violations Found in KubeCost-Gym v3.0 → v3.1

### Audit Fix 01: Float Equality in ColdStartGrader

- **Bug:** `http_error_rate == 0.0` fails when rate=1e-15 (floating-point arithmetic)
- **Fix:** `http_error_rate < 0.001` with tolerance
- **Lesson:** Contract must specify tolerance, not bare equality

### Audit Fix 02: Unnormalized EfficientSqueezeGrader

- **Bug:** `score -= 0.05` per violation, unbounded; different-length traces get different scores for same violation rate
- **Fix:** `score = 1.0 - (violations / len(trajectory))`
- **Lesson:** Grader spec must explicitly state "normalized by trace length"

### Audit Fix 03: Sparse Reward Cliff

- **Bug:** Binary reward at p99=300ms; zero gradient everywhere else; agents cannot learn latency awareness
- **Fix:** Linear ramp penalty for p99 ∈ [200, 300ms)
- **Lesson:** Reward spec must label every threshold as cliff or ramp

### Audit Fix 04: Unsolvable Hard Task

- **Bug:** Hard task requires absorbing +20 replicas in 1 step; max available action was +10; structurally unsolvable
- **Fix:** Add `SCALE_REPLICAS(+20)` to ActionType enum
- **Lesson:** Every task must be proven solvable by available action sequences

### Audit Fix 05: Wrong Hardware Tag

- **Bug:** `hardware: cpu-upgrade` allocates 8 vCPU; hackathon constraint is ≤2 vCPU
- **Fix:** `hardware: cpu-basic` (2 vCPU / 8 GB RAM)
- **Lesson:** Infra constraints are part of the spec; map explicitly

---

## 6. The Three Tasks

### Task 1: Cold Start (Easy)

**Objective:** Scale cluster from 0 replicas to 5 in minimum steps without SLA breach.

- **Difficulty:** Reactive solution works
- **Success Signal:** Immediate (replicas ≥ 5, error_rate ≈ 0)
- **Grader:** `ColdStartGrader` — normalizes by: 1.0 - (error_rate)

### Task 2: Efficient Squeeze (Medium)

**Objective:** Maintain sub-20% stealth across a full 24-hour sinusoidal load cycle while minimizing cost.

- **Difficulty:** Requires holding state and trend-following
- **Success Signal:** Sustained (density maintained, cost optimal)
- **Grader:** `EfficientSqueezeGrader` — normalizes by: 1.0 - (violations / trace_length)

### Task 3: Entropy Storm (Hard)

**Objective:** Issue `REBALANCE_NODE` BEFORE steal exceeds 20%, using only leading indicators.

- **Difficulty:** Failure is latent; SLA signal arrives too late to fix reactively
- **Success Signal:** Proactive action before breach
- **Grader:** `EntropyStormGrader` — proactive action bonus only when prediction is correct

---

## 7. Pre-Submission Checklist

### Automated Validator Gates (Disqualifying)

- [ ] HF Space URL returns HTTP 200 and responds to `reset()`
- [ ] `docker build` succeeds with zero errors from clean directory
- [ ] `inference.py` exists in root directory (not subdirectory)
- [ ] `inference.py` runs end-to-end in <20 minutes
- [ ] `openenv.yaml` parses without error; has name, version, description, tasks[]
- [ ] Each task has name, difficulty (easy/medium/hard), description
- [ ] All 3 graders return float in [0.0, 1.0]
- [ ] README.md has `hardware: cpu-basic`, `sdk: docker`, `tags: - openenv`

### Spec Compliance Checks

- [ ] No stub bodies (`pass`) remain in `env.py`
- [ ] `state()` returns typed Pydantic model, not bare dict
- [ ] No float equality (`== 0.0`) — all replaced with tolerance checks
- [ ] All graders handle empty trajectory: `if not trajectory: return 0.0`
- [ ] Reward function has no cliff at SLA threshold — ramp present
- [ ] Every hard task is solvable by at least one action sequence

### Documentation Checks

- [ ] All README citations verified to exist and claim correctly
- [ ] Observation space dimensions documented (name, type, range)
- [ ] Action space documented with plain-English description per action
- [ ] Setup instructions complete: Python version, install, env vars, run command

---

## 8. Common Mistakes to Avoid

| Mistake                     | Why It Fails                                      | Prevention                                                              |
| --------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| Stubs left as `pass`        | `step()` returns None; tuple unpacking crashes    | Search for `pass`; all bodies must return correct type                  |
| Grader score > 1.0          | Unbounded accumulation; validator rejects         | Always normalize by `len(trajectory)`; clamp with `max(0, min(1, raw))` |
| Wrong hardware tag          | `cpu-upgrade` exceeds 2 vCPU limit                | Use `hardware: cpu-basic` exclusively                                   |
| inference.py not in root    | Validator looks for `./inference.py` specifically | Never move to subdirectory                                              |
| Float equality              | Floating point never equals exact zero            | Replace `== 0.0` with `< 1e-3`                                          |
| Binary reward cliff         | Agent cannot learn latency awareness              | Add linear penalty in warning zone before threshold                     |
| Unsolvable hard task        | No action sequence can produce required outcome   | Verify solvability before writing grader                                |
| Fabricated citations        | Judges verify sources; instant integrity flag     | Only cite sources you have personally read                              |
| Variable-length observation | ML agents expect fixed-dim input                  | Zero-pad all lists to fixed max; declare `min_length = max_length`      |

---

## 9. Judging Criteria

| Criterion            | What Judges Evaluate                                                                  | SDD Phase                            | Weight |
| -------------------- | ------------------------------------------------------------------------------------- | ------------------------------------ | ------ |
| Runtime Correctness  | Runs without errors; correct types; inference completes                               | Phase 2 (Contract) + Phase 5 (Infra) | Gate   |
| Interface Compliance | Follows OpenEnv standard; typed models; no bare dicts                                 | Phase 2 (Contract)                   | High   |
| Task Design          | Real-world problem; genuine difficulty ladder; hard task requires proactive reasoning | Phase 1 (Domain)                     | High   |
| Grading Logic        | Sensible reward; normalized graders; no sparse cliffs; proactivity rewarded           | Phase 3 (Reward) + Phase 4 (Grader)  | High   |

**Prioritization:** Runtime is gate. After passing, invest most remaining effort in task design quality and reward signal design.

---

## 10. SDD Workflow — 8 Steps from Idea to Submission

```
Step 1: Write domain-spec.md
  - What real-world system? Observable/controllable vars?
  - Task ladder: easy (reactive) → medium (balance) → hard (proactive)

Step 2: Write Pydantic models (no logic)
  - Observation: all fields with types and Field() constraints
  - ActionType: complete enum
  - EnvState: typed return type

Step 3: Write reward formula as math, THEN implement
  - No cliffs without ramps
  - Bound the range: R_min, R_max
  - Proactive bonuses

Step 4: Write grader formulas as math, THEN implement
  - Normalize by len(trajectory)
  - Final clamp: max(0.0, min(1.0, raw))
  - Test: empty, all violations, no violations

Step 5: Implement environment body (no stubs)
  - reset(), step(), state(), _apply_action(), _calculate_reward()
  - Run validate_local.py; assert all checks pass

Step 6: Write inference.py
  - Root directory
  - Read API_BASE_URL, MODEL_NAME, HF_TOKEN exactly
  - OpenAI client + JSON mode

Step 7: Configure infrastructure
  - Dockerfile: python:3.10-slim, requirements.txt first
  - README: hardware: cpu-basic, sdk: docker, tags: - openenv

Step 8: Verify and submit
  - Run validate_local.py
  - docker build . (succeeds)
  - Push to HF Space
  - Confirm Space running
  - Submit URL before deadline
```

---

## 11. Local Validation Script

```python
from env import KubeCostEnv, Action, ActionType, Observation, EnvState
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader

def validate():
    env = KubeCostEnv("traces/trace_v1_coldstart.json")

    # 1. reset() returns typed Observation
    obs = env.reset()
    assert isinstance(obs, Observation)

    # 2. state() returns typed model (not dict)
    state = env.state()
    assert isinstance(state, EnvState)

    # 3. step() returns correct 4-tuple
    action = Action(action_type=ActionType.MAINTAIN)
    obs2, reward, done, info = env.step(action)
    assert isinstance(obs2, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    # 4. All graders return [0.0, 1.0]
    trajectory = [{"observation": obs2, "uptime_metric": 1.0, "cost_metric": 0.8}]
    for grader in [ColdStartGrader(), EfficientSqueezeGrader(), EntropyStormGrader()]:
        score = grader.grade(trajectory)
        assert 0.0 <= score <= 1.0, f"{grader.__class__.__name__}: score={score} out of range"
        print(f"  {grader.__class__.__name__}: {score:.3f} ✓")

    print("All validation checks passed.")

if __name__ == "__main__":
    validate()
```

---

## 12. Key Takeaways

1. **Specification comes first** — Domain, Contract, Reward, Grader, Infra. Write each spec before implementing.
2. **Normalize everything** — Graders must return [0.0, 1.0] regardless of trajectory length.
3. **No sparse cliffs in rewards** — Every hard threshold needs a gradient ramp.
4. **Contract prevents bugs** — Pydantic models with Field() constraints catch spec violations at runtime.
5. **Task design matters most** — Judges value genuine difficulty progression and proactive reasoning.
6. **Infrastructure is part of the spec** — Hardware tags, YAML structure, and file placement are validated by automata.
7. **Verify solvability** — Every task (especially hard) must be provably solvable by available actions.
8. **Determinism is reproducibility** — Use pre-recorded traces, not RNG, for validation consistency.

---

**Last Updated:** March 30, 2026  
**Reference Implementation:** KubeCost-Gym v3.1 (21/30 → 29/30 via SDD)  
**Framework Used:** OpenEnv + HuggingFace Spaces + Python 3.10+
