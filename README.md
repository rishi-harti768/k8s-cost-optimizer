---
title: KubeCost-Gym
emoji: ⚙️
colorFrom: blue
colorTo: purple
sdk: docker
hardware: cpu-basic
tags:
  - openenv
  - reinforcement-learning
  - kubernetes
  - cost-optimization
---

# KubeCost-Gym v3.1: Kubernetes Cost Optimization RL Environment

A production-grade RL environment for learning proactive Kubernetes autoscaling strategies using LLMs. Implements the **OpenEnv** standard with strict Pydantic type validation, deterministic trace-based dynamics, and three progressive difficulty tasks testing reactive, balancing, and proactive decision-making.

**Deadline:** 8 Apr 2026, 11:59 PM IST  
**Target Score:** ≥27/30  
**Framework:** OpenEnv · openenv-core FastAPI · HuggingFace Spaces · Python 3.10+

---

## 🚀 Quick Start: Deploy to HuggingFace Spaces

**Ready to deploy?** See [**DEPLOYMENT.md**](DEPLOYMENT.md) for complete step-by-step instructions.

**Quick checklist:**
- [ ] Create a HuggingFace Space with Docker SDK and `cpu-basic` hardware
- [ ] Set `HF_TOKEN` secret (your LLM API key)
- [ ] Optionally set `API_BASE_URL` and `MODEL_NAME` if using non-default LLM
- [ ] Push this repository to your Space
- [ ] Access the REST API at `https://<username>-kubecost-gym.hf.space` (with `/docs` for interactive Swagger UI)

---

## Overview

This environment simulates a production Kubernetes cluster facing:

- **Dynamic workloads** (sinusoidal CPU/memory curves)
- **Resource constraints** (node capacity, cost budget)
- **SLA targets** (p99 latency < 300ms, error rate minimization)

The agent must learn to:

1. Scale replicas up/down in response to load
2. Balance cost vs. reliability tradeoffs
3. Predict and prevent problems using leading indicators (CPU steal)

---

## The Three Tasks

| Task                  | Difficulty | Objective                                 | Success Signal                           |
| --------------------- | ---------- | ----------------------------------------- | ---------------------------------------- |
| **Cold Start**        | Easy       | Scale 0→5 replicas without SLA breach     | Immediate (replicas ≥5, error_rate ≈0)   |
| **Efficient Squeeze** | Medium     | Maintain <20% steal over 24h load cycle   | Sustained (density held, cost optimized) |
| **Entropy Storm**     | Hard       | Proactive REBALANCE_NODE before steal>20% | Predictive (correct before breach)       |

**Key Insight:** Hard task requires proactive reasoning with leading indicators—reactive agents cannot recover from SLA breaches in time.

---

## Observation Space

All fields are continuously available to the agent:

| Field                 | Type        | Range     | Purpose                                       |
| --------------------- | ----------- | --------- | --------------------------------------------- |
| `cpu_usage_pct`       | float       | [0, 100]  | Cluster-wide CPU utilization                  |
| `mem_usage_pct`       | float       | [0, 100]  | Cluster-wide memory utilization               |
| `p99_latency_ms`      | float       | [0, ∞)    | Tail latency (SLA: 300ms)                     |
| `http_error_rate`     | float       | [0, 1]    | Request failure rate                          |
| `cpu_steal_pct`       | float       | [0, 1]    | Noisy-neighbor indicator (leading signal)     |
| `active_replicas`     | int         | [0, ∞)    | Running pod count                             |
| `buffer_depth`        | int         | [0, ∞)    | Request queue depth                           |
| `node_size_class`     | Enum        | {S, M, L} | Node tier (Small/Medium/Large)                |
| `current_hourly_cost` | float       | [0, ∞)    | USD/hour spend                                |
| `node_bin_density`    | List[float] | [0, 1]×10 | Per-node packing ratio (fixed 10-elem vector) |

---

## Action Space

The agent selects from 9 discrete actions:

| Action                | Effect                  | Use Case                                       |
| --------------------- | ----------------------- | ---------------------------------------------- |
| `SCALE_REPLICAS(-5)`  | Aggressive scale-down   | Cost reduction when surge over                 |
| `SCALE_REPLICAS(-1)`  | Gentle scale-down       | Gradual resource release                       |
| `MAINTAIN`            | Hold current state      | No action needed                               |
| `SCALE_REPLICAS(+1)`  | Gentle scale-up         | Incremental capacity                           |
| `SCALE_REPLICAS(+5)`  | Moderate scale-up       | Standard load increase                         |
| `SCALE_REPLICAS(+10)` | Large scale-up          | Major surge handling                           |
| `SCALE_REPLICAS(+20)` | Emergency burst         | Crisis capacity (hard task)                    |
| `UPGRADE_NODE`        | Vertical scale          | Move to larger node tier (irreversible 1-step) |
| `REBALANCE_NODE`      | Deterministic rebalance | Fix noisy-neighbor issues (hard task)          |

---

## Known Limitations & Troubleshooting

### Critical Fixes Applied (Phase 4 Remediation)

This codebase has been aggressively reviewed and fixed for the following critical issues:

#### 1. Physics Overlay Removed (Fix #1)

✅ **Status:** Fixed

- **Issue:** Observations were being modified by formula overlays, breaking determinism
- **Resolution:** Observations now come directly from traces with NO modifications
- **Impact:** All traces are now ground truth; reproducibility guaranteed

#### 2. Cost Penalty Bounded (Fix #2)

✅ **Status:** Fixed

- **Issue:** Cost penalty could exceed 5.0 in reward formula, breaking bounds
- **Resolution:** Cost penalty now clamped: `min(5.0, 5.0 * cost_fraction)`
- **Impact:** Reward strictly bounded to [-20.0, +10.5]

#### 3. Proactive Bonus Timing Fixed (Fix #3)

✅ **Status:** Fixed

- **Issue:** Previous steal comparison was off-by-one
- **Resolution:** Steal snapshot captured after step advances, before reward calc
- **Impact:** Task 3 learning signal now correct

#### 4. EntropyStormGrader Logic Fixed (Fix #4)

✅ **Status:** Fixed

- **Issue:** Passive agents could score 1.0 with no violations
- **Resolution:** Requires REBALANCE actions for perfect score when no violations occur
- **Impact:** Prevents reward hacking; encourages proactive behavior

#### 5. Episode Termination Fixed (Fix #5)

✅ **Status:** Fixed

- **Issue:** Episodes didn't terminate at trace end, causing duplicate observations
- **Resolution:** Early exit when `_step >= total_steps`
- **Impact:** No more wrapped observations in Task 2 & 3 grading

#### 6. Pydantic Config Standardized (Fix #9)

✅ **Status:** Fixed

- **Issue:** Inconsistent `use_enum_values` across models
- **Resolution:** All models now use `use_enum_values = True`
- **Impact:** Consistent JSON serialization; LLM response parsing more reliable

#### 7. LLM Response Parsing Enhanced (Fix #10)

✅ **Status:** Fixed

- **Issue:** Couldn't handle markdown code blocks, failed on name vs. value mismatches
- **Resolution:** Robust parsing with markdown handling, name/value fallback, validation
- **Impact:** Inference more reliable even with LLM formatting variations

#### 8. Environment Validation Enhanced (Fix #12)

✅ **Status:** Fixed

- **Issue:** Env vars checked for existence but not format
- **Resolution:** URL format validation, token length checks
- **Impact:** Configuration errors caught early with clear messages

#### 9. Inference Main Completed (Fix #13)

✅ **Status:** Fixed

- **Issue:** Main function incomplete, missing result output
- **Resolution:** Full results collection, file output, exit codes
- **Impact:** Inference pipeline is production-ready

#### 10. Empty Trajectory Error Logging (Fix #8)

✅ **Status:** Fixed

- **Issue:** Empty trajectories silently returned 0.0
- **Resolution:** Error logging in graders, trajectory validation in inference
- **Impact:** Failures surface immediately instead of being silent

### Performance Considerations

#### Traces Are Minimal Fixtures

⚠️ **Note:** Current traces (5-6 steps) are for rapid testing only.

**For production:**

- Traces should be 50+ steps (realistic time horizons)
- Should be recorded from real K8s clusters or high-fidelity simulators
- Should include diverse failure modes (cascades, oscillations, network failures)

#### LLM Integration Constraints

- **Timeout:** 30 seconds per LLM decision (prevents hangs)
- **Fallback:** If LLM fails to parse: defaults to MAINTAIN (safe)
- **Tested:** OpenAI API only; other providers may require adapter code

### Design Notes

#### Observation Immutability

Observations are taken DIRECTLY from traces with NO formula overlays:

- Agent actions (SCALE_REPLICAS, etc.) do NOT modify observations in the current step
- Consequences are modeled in future trace steps (or can be)
- To enable action feedback in observations, regenerate traces with action-dependent dynamics

#### Task-Specific Scoring

**Task 1: Cold Start (Easy)**

- **Metric:** Average HTTP error rate
- **Score:** `1.0 - avg_error_rate`
- **Pass:** Score > 0.8

**Task 2: Efficient Squeeze (Medium)**

- **Metric:** CPU steal violations (steal_pct >= 0.20)
- **Score:** `1.0 - (violations / len(trajectory))`
- **Pass:** Score > 0.7

**Task 3: Entropy Storm (Hard)**

- **Metric:** Proactive REBALANCE actions within 5 steps before steal violation
- **Score:** `proactive_actions / total_violations` (or 0.0 if no violations and passive)
- **Pass:** Score > 0.6

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for full scoring formulas.

### Running Tests

```bash
# Tests are included in uv dev dependencies (automatically installed with uv sync --dev)
# uv automatically manages all test dependencies

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_graders.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html
```

### Troubleshooting Common Issues

**Q: Grader returns 0.0 on every episode**
A: Check that trajectory is non-empty. If still 0.0, ensure observations have valid values. Enable stderr to see error logs.

**Q: LLM inference hangs**
A: Check API_BASE_URL is correct and accessible. Default timeout is 30s; increase if needed.

**Q: EntropyStormGrader scores < 0.5 even with REBALANCE actions**
A: Verify REBALANCE was issued within 5-step lookback window BEFORE steal violation appears.

**Q: Cost penalty seems high**
A: Cost is clamped at 5.0 per step. If hourly_cost > BUDGET, penalty = 5.0. Check trace cost values.

---

# Sync project environment (creates .venv automatically)

uv sync

# Set environment variables (REQUIRED - see Pre-Submission Checklist below)

export API*BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="hf*..." # Your HuggingFace token

# Run local validation

python validate_local.py

# Run inference pipeline

python inference.py

````

### Docker Deployment

```bash
# Build image
docker build -t kubecost-gym .

# Run container with required env vars
docker run \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="hf_..." \
  kubecost-gym
````

### HuggingFace Spaces Deployment

Push to HF Space with Docker SDK:

```bash
# Create Space: https://huggingface.co/new-space
# Select: Docker SDK, cpu-basic hardware

git clone https://huggingface.co/spaces/<user>/<space-name>
cd <space-name>
git remote add origin <space-repo>

# Copy files
cp -r ../ ./

# Setup Secrets in Space UI:
# - API_BASE_URL
# - MODEL_NAME
# - HF_TOKEN (validator uses this)

# Push to deploy
git add .
git commit -m "Initial commit"
git push origin main
```

---

## Pre-Submission Checklist

**CRITICAL:** Before hitting "Submit" on the assessment link, verify these items or you will be disqualified.

### ✓ Environment Variables (Must Pass)

All three required variables must be set and use `os.environ.get()`:

```python
# ✓ CORRECT
api_base_url = os.environ.get("API_BASE_URL")
model_name = os.environ.get("MODEL_NAME")
hf_token = os.environ.get("HF_TOKEN")

# ✗ WRONG (hardcoded)
api_base_url = "https://api.openai.com/v1"  # ← DISQUALIFIED
hf_token = "hf_abc123..."  # ← DISQUALIFIED
```

| Variable       | Required | Example                       | Purpose                      |
| -------------- | -------- | ----------------------------- | ---------------------------- |
| `API_BASE_URL` | **YES**  | `https://api.openai.com/v1`   | The API endpoint for the LLM |
| `MODEL_NAME`   | **YES**  | `gpt-4` or provider-specific  | The model identifier to use  |
| `HF_TOKEN`     | **YES**  | `hf_...` (validator provides) | HuggingFace / API token      |

### ✓ Inference Script Location (Must Pass)

```
✓ Accepted:
  k8s-cost-optimizer/inference.py

✗ REJECTED (hidden in subdirectory):
  k8s-cost-optimizer/scripts/inference.py
  k8s-cost-optimizer/src/inference.py
```

**Automated grader will fail immediately if not in root.**

### ✓ Grader Output Bounds (Must Pass)

All graders must return scores strictly in [0.0, 1.0]:

```python
# ✓ CORRECT (clamped)
def grade(self, trajectory):
    score = calculate_score(trajectory)
    return max(0.0, min(1.0, score))  # ← Always clamp

# ✗ WRONG (unbounded)
def grade(self, trajectory):
    return 1.5  # ← FAILS validation
    # or
    score = 1.0
    score -= 0.05  # per something ← Unbounded!
    return score
```

**Double-check in graders.py:**

- `ColdStartGrader.grade()` returns [0.0, 1.0]
- `EfficientSqueezeGrader.grade()` returns [0.0, 1.0]
- `EntropyStormGrader.grade()` returns [0.0, 1.0]

### ✓ OpenAI Client Usage (Must Pass)

All LLM calls must use OpenAI Client (not Google Gemini):

```python
# ✓ CORRECT
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("HF_TOKEN"), base_url=api_base_url)
response = client.chat.completions.create(model=model_name, messages=[...])

# ✗ WRONG (Google Gemini)
import google.generativeai as genai  # ← DISQUALIFIED
model = genai.GenerativeModel("gemini-2.5-flash")
```

### ✓ 3+ Graders Present (Must Pass)

All three tasks must have working graders:

```python
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader

# ✓ All three defined and callable
tasks = [
    ("cold_start", ColdStartGrader()),
    ("efficient_squeeze", EfficientSqueezeGrader()),
    ("entropy_storm", EntropyStormGrader()),
]
```

### ✓ OpenEnv Spec Compliance (Must Pass)

Validate `openenv.yaml` structure:

```yaml
name: kubecost-gym
version: "3.0" # ← String, not float!
description: "..."
tasks:
  - name: cold_start
    difficulty: easy
    description: "..."
  - name: efficient_squeeze
    difficulty: medium
    description: "..."
  - name: entropy_storm
    difficulty: hard
    description: "..."
```

### ✓ Dockerfile Builds (Must Pass)

```bash
docker build -t kubecost-gym .
# Must complete without errors
```

Dockerfile must:

- Build from `python:3.10-slim`
- Copy `inference.py` from root
- Verify `inference.py` exists: `test -f inference.py || exit 1`

### ✓ Baseline Reproducibility (Must Pass)

Run submitted inference script—must complete without error:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="hf_xyz..."
python inference.py
# Exit code: 0 (success)
# No unhandled exceptions
```

### ✓ Infra Requirements (Must Pass)

- **Runtime:** < 20 minutes on vcpu=2, memory=8gb
- **No hardcoded credentials:** Use `os.environ.get()`
- **Stateless inference:** No side effects, reproducible

### ✓ Run Local Validator

```bash
python validate_local.py
# All checks must PASS (green ✓)
```

This validates:

- Syntax errors
- Environment variable patterns
- Grader bounds
- OpenAI Client usage
- File locations
- Dockerfile structure

---

## Environment Variables

---

## OpenEnv Interface

The environment implements the **OpenEnv standard**:

```python
from env import KubeCostEnv
from models import Action, ActionType, Observation, EnvState

# Initialize
env = KubeCostEnv("traces/trace_v1_coldstart.json")

# Reset
obs: Observation = env.reset()  # Returns typed model, not dict
print(obs.cpu_usage_pct)  # 45.0
print(obs.node_bin_density)  # [0.3, 0.4, ...]

# Step
action = Action(action_type=ActionType.SCALE_UP_5)
obs_next, reward, done, info = env.step(action)  # 4-tuple
assert isinstance(reward, float)
assert isinstance(done, bool)

# State
state: EnvState = env.state()  # Typed model, not dict
print(state.step, state.replicas, state.node_size)
```

---

## Project Structure

```
k8s-cost-optimizer/
│
├── openenv.yaml              ← Task definitions & metadata
├── models.py                 ← Pydantic type definitions
├── env.py                    ← KubeCostEnv (OpenEnv interface)
├── graders.py                ← Three task graders
├── inference.py              ← LLM agent (in root)
│
├── server/
│   └── app.py                ← FastAPI REST server (openenv-core)
│
├── pyproject.toml            ← Project metadata and dependencies
├── uv.lock                   ← Locked dependency versions (universal)
├── Dockerfile                ← Docker image spec
├── README.md                 ← This file
├── validate_local.py         ← Pre-submission validator
│
├── traces/                   ← Deterministic dynamics (JSON)
│   ├── trace_v1_coldstart.json
│   ├── trace_v1_squeeze.json
│   └── trace_v1_entropy.json
│
└── PROJECT_SPEC.md           ← Complete SDD requirements
```

---

## Spec-Driven Development (SDD)

This project implements all 5 SDD phases:

### Phase 1: Domain Specification

- Real-world problem: K8s autoscaling balances cost vs. reliability
- Observable: CPU/memory/latency/steal metrics, pod count, cost
- Controllable: Replica scaling, node upgrades, rebalancing
- Task ladder: reactive → balanced → proactive

### Phase 2: Contract Specification

- Pydantic models with Field() constraints
- No bare strings—all finite sets use Enum
- Fixed-length vectors: node_bin_density[10]
- state() returns EnvState (typed), not dict

### Phase 3: Reward Specification

- No sparse cliffs: linear ramp penalty for p99 ∈ [200, 300ms)
- Bounded: R ∈ [-20.0, +10.5]
- Competing objectives: Uptime AND Cost terms
- Proactive bonuses for early corrective action

### Phase 4: Grader Specification

- Normalized: score = 1.0 - (violations / len(trajectory))
- No unbounded accumulation (never -= 0.05)
- Float tolerance: use < 0.001, never == 0.0
- Clamp: max(0.0, min(1.0, raw_score))

### Phase 5: Infrastructure Specification

- Dockerfile: python:3.10-slim base
- inference.py: root directory, reads env vars, runs <20 min
- README: hardware: cpu-basic (2 vCPU / 8 GB max)
- openenv.yaml: version as string ("3.0"), not float

---

## Validation

Run pre-submission checks **before** submitting:

```bash
python validate_local.py
```

This validates the critical gates:

- ✓ All modules import without syntax errors
- ✓ openenv.yaml parses and has required fields (3 tasks)
- ✓ All graders enforce [0.0, 1.0] bounds
- ✓ Environment variables use `os.environ.get()`
- ✓ OpenAI Client imported (not Google Gemini)
- ✓ inference.py exists in root directory
- ✓ pyproject.toml has OpenAI (not google-generativeai)
- ✓ Dockerfile includes critical checks
- ✓ No hardcoded credentials detected
- ✓ Traces directory ready

**Must pass all checks before submission** or you will be disqualified.

---

## Audit Fixes (v3.0 → v3.1)

This implementation incorporates all 5 audit fixes from the specification:

| Fix    | Issue                                 | Resolution                                        |
| ------ | ------------------------------------- | ------------------------------------------------- |
| **01** | Float equality in ColdStartGrader     | Use tolerance: `< 0.001`, not `== 0.0`            |
| **02** | Unbounded grader (score -= 0.05)      | Normalize: `1.0 - (violations / len(trajectory))` |
| **03** | Sparse reward cliff at p99=300ms      | Add linear ramp penalty [200, 300ms)              |
| **04** | Hard task unsolvable (no +20 scaling) | Add `SCALE_REPLICAS(+20)` enum value              |
| **05** | Wrong hardware tag (cpu-upgrade)      | Use `hardware: cpu-basic` only                    |

---

## Common Mistakes to Avoid

| Mistake                               | Impact                  | Prevention                                        |
| ------------------------------------- | ----------------------- | ------------------------------------------------- |
| inference.py in subdirectory          | **DISQUALIFIED**        | Keep in root, never move                          |
| Hardcoded API key/token               | **DISQUALIFIED**        | Always use `os.environ.get(key)`                  |
| Grader score > 1.0 or < 0.0           | **DISQUALIFIED**        | Normalize by trajectory length + clamp            |
| Using Google Gemini instead of OpenAI | **DISQUALIFIED**        | `from openai import OpenAI`                       |
| Wrong environment variable names      | **DISQUALIFIED**        | Must be `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`  |
| Float equality `== 0.0`               | Fails edge cases        | Use `< tolerance` always                          |
| Sparse reward cliff                   | No gradient             | Add linear ramp before threshold                  |
| Unsolvable hard task                  | Structurally impossible | Verify action space complete                      |
| Unbounded grader accumulation         | Invalid scores (>1.0)   | Normalize: `1.0 - (violations / len(trajectory))` |

---

## Judging Criteria

| Criterion                | Weight | Pass/Fail                                      |
| ------------------------ | ------ | ---------------------------------------------- |
| **Runtime Correctness**  | Gate   | Runs without errors, correct types             |
| **Interface Compliance** | High   | Typed models, OpenEnv standard                 |
| **Task Design**          | High   | Genuine difficulty ladder, proactive reasoning |
| **Grading Logic**        | High   | Normalized, no cliffs, sensible reward         |

---

## Key SDD Takeaways

1. **Specification first** — Each phase (Domain → Contract → Reward → Grader → Infra) completed before implementation
2. **Type safety** — Pydantic models prevent spec violations at runtime
3. **Normalization** — All graders return [0.0, 1.0] regardless of trajectory length
4. **No sparse cliffs** — Every hard reward threshold has a gradient ramp
5. **Solvability proof** — Every task (especially hard) verified achievable by available actions
6. **Reproducibility** — Deterministic traces, no RNG, ensures consistent evaluation

---

## References

- **Project Specification:** [PROJECT_SPEC.md](PROJECT_SPEC.md)
- **OpenEnv Standard:** [openenv.yaml](openenv.yaml)
- **Local Validation:** [validate_local.py](validate_local.py)
- **SDD Methodology:** See spec §3 and §10

---

## Deadline & Submission

- **Deadline:** 8 Apr 2026, 11:59 PM IST
- **Target Score:** ≥27/30 (28/30 from task design + grading quality)
- **Submit:** HuggingFace Space URL
- **Pre-flight:** Run `validate_local.py` before submitting

---

**Last Updated:** March 30, 2026  
**Status:** Implementation Phase (Core Logic Complete)  
**Implementation Phase:** Phase 5 Infrastructure Specification (Next)

| Phase | Specification           | Status      | Description                                |
| ----- | ----------------------- | ----------- | ------------------------------------------ |
| 1     | **Domain Spec**         | ✅ Complete | Task ladder & OpenEnv configuration        |
| 2     | **Contract Spec**       | ✅ Complete | Pydantic models & validation constraints   |
| 3     | **Reward Spec**         | ✅ Complete | Dense reward formula & gradient ramp       |
| 4     | **Grader Spec**         | ✅ Complete | Normalized, length-invariant scoring logic |
| 5     | **Infrastructure Spec** | 🚀 Next     | Docker, Inference, & HF Space Deployment   |
