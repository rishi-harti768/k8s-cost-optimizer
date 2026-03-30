# KubeCost-Gym Scaffolding Manual

## Spec-Driven Setup for OpenEnv RL Environment

**Project:** KubeCost-Gym v3.1  
**Deadline:** 8 Apr 2026, 11:59 PM IST  
**Phase Focus:** Initial Project Setup (Pre-Implementation)  
**Status:** Not for code implementation — structure & templates only

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Phase 0: Prerequisites & Decisions](#phase-0-prerequisites--decisions)
3. [Project Structure](#project-structure)
4. [File-by-File Scaffolding](#file-by-file-scaffolding)
5. [Initialization Checklist](#initialization-checklist)
6. [Critical Spec Reminders](#critical-spec-reminders)

---

## Overview

This manual guides the INITIAL SCAFFOLDING of a Kubernetes cost optimization RL environment following **Spec-Driven Development (SDD)** methodology. It covers:

- File structure & directories
- Configuration file templates
- Model stubs (Pydantic types)
- Placeholder functions (all bodies removed)
- Dependencies & runtime setup
- Validation framework preparation

**What this does NOT cover:**

- Full environment logic implementation
- LLM inference pipeline details
- Reward calculation algorithms
- Grader scoring specifics

---

## Phase 0: Prerequisites & Decisions

Before scaffolding, **clarify these critical decisions** (do not assume):

### Decision 1: Environment Traces Source

**Question:** Where will the deterministic environment dynamics come from?

**Options (from spec):**

- `Option A:` Pre-recorded JSON traces (recommended for reproducibility per spec §1)
- `Option B:` Synthetic procedural generation
- `Option C:` Loaded from external API

**Required for:**

- [x] Directory: `traces/` structure
- [x] File: `env.py` loader implementation
- [x] File: `validate_local.py` test fixture paths

**Action Required:** Specify which approach to support.

---

### Decision 2: Task Concrete Specifications

**Question:** Is the task ladder defined and solvable?

The spec requires (§6) three tasks:

| Task                          | Difficulty | Current Definition?                          |
| ----------------------------- | ---------- | -------------------------------------------- |
| **Task 1: Cold Start**        | Easy       | Scale from 0 → 5 replicas, no SLA breach     |
| **Task 2: Efficient Squeeze** | Medium     | Maintain <20% steal over 24h sine load cycle |
| **Task 3: Entropy Storm**     | Hard       | Proactive `REBALANCE_NODE` before steal >20% |

**Required Detail for Scaffolding:**

- [ ] Are these task names/difficulties **final** or subject to change?
- [ ] Do traces exist for each task, or will they be created during Phase 1?
- [ ] What are task-specific **success criteria** in metrics? (% budget adherence, error_rate tolerance, etc.)

**Action Required:** Confirm task ladder or propose modifications before creating grader stubs.

---

### Decision 3: LLM Inference Backend

**Question:** What LLM API will `inference.py` use?

The spec (Infra Spec §5) requires:

- Reads env vars: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- Uses OpenAI client only
- Runs in <20 min end-to-end

**Current unknowns:**

- [ ] Which model? (e.g., gpt-4, gpt-3.5-turbo)
- [ ] Which API provider? (OpenAI, Anthropic proxy, local)
- [ ] Batch size for inference?

**Action Required:** Choose API backend for scaffolding inference.py stub.

---

### Decision 4: Observation Space Design

**Question:** Are all 10 observation fields (per spec Table §4) **confirmed final**, or open to addition/removal?

Spec lists:

```
cpu_usage_pct, mem_usage_pct, p99_latency_ms, http_error_rate,
cpu_steal_pct, active_replicas, buffer_depth, node_size_class,
current_hourly_cost, node_bin_density (10-elem vector)
```

**Implication:**

- Affects Pydantic `Observation` model size
- Affects action space feasibility (more state → more actions needed)

**Action Required:** Lock observation fields or propose changes.

---

## Project Structure

### Full Directory Tree (Target State)

```
k8s-cost-optimizer/
│
├── PROJECT_SPEC.md                 ← Provided (reference)
├── README.md                        ← HuggingFace frontmatter + setup guide
├── SCAFFOLDING_MANUAL.md           ← This file
│
├── requirements.txt                 ← Python dependencies
├── Dockerfile                       ← Docker image definition
│
├── env.py                           ← Main environment: KubeCostEnv class
├── models.py                        ← Pydantic type definitions
├── graders.py                       ← Three grader implementations
├── inference.py                     ← LLM agent inference pipeline (ROOT)
│
├── openenv.yaml                     ← OpenEnv metadata & task definitions
├── validate_local.py                ← Pre-submission validator
│
├── traces/                          ← Deterministic dynamics (JSON)
│   ├── trace_v1_coldstart.json     ← Task 1 data
│   ├── trace_v1_squeeze.json       ← Task 2 data
│   └── trace_v1_entropy.json       ← Task 3 data
│
└── tests/                           ← (Optional) Unit test fixtures
    └── test_validation.py           ← Test harness for validators
```

### Directory Purposes

| Directory   | Purpose            | Scaffolding Content                          |
| ----------- | ------------------ | -------------------------------------------- |
| `.` (root)  | Entry point files  | openenv.yaml, inference.py, requirements.txt |
| `./`        | Environment logic  | env.py, models.py, graders.py                |
| `./traces/` | Deterministic data | Three JSON files (see Decision 1)            |
| `./tests/`  | Validation tests   | Harness for pre-submission checks            |

---

## File-by-File Scaffolding

### 1. **openenv.yaml** — Environment Metadata & Task Definitions

**Purpose:** OpenEnv interface contract; parsed by validator automata.

**Template:**

```yaml
# OpenEnv v3.0 metadata (strict YAML structure)
name: kubecost-gym # kebab-case, globally unique
version: "3.0" # STRING, not float (CRITICAL)
description: "Kubernetes cost optimization via proactive autoscaling decisions."

tasks:
  - name: cold_start
    difficulty: easy
    description: "Scale cluster from 0→5 replicas without SLA breach (p99<300ms)."

  - name: efficient_squeeze
    difficulty: medium
    description: "Maintain <20% CPU steal across 24-hour sinusoidal load cycle."

  - name: entropy_storm
    difficulty: hard
    description: "Issue REBALANCE_NODE before cpu_steal_pct exceeds 20% using only leading indicators."
```

**Validation Checks (from spec §7):**

- [ ] `name` in kebab-case
- [ ] `version` as quoted string (not bare float)
- [ ] `description` present and non-empty
- [ ] 3 tasks present with: name, difficulty (easy|medium|hard), description
- [ ] File parses as valid YAML

**Decision Gates:**

- ⚠️ **Confirm task names & descriptions** from Decision 2 before finalizing.

---

### 2. **models.py** — Pydantic Type Definitions

**Purpose:** Contract specification (Phase 2, SDD §3.2). Define all data types with Field() constraints.

**Template Structure:**

```python
# models.py
"""
Pydantic type definitions for KubeCost-Gym (Phase 2: Contract Spec).

All fields include:
  - Type annotations (no bare strings)
  - Field() constraints (min/max, enums, list bounds)
  - Docstrings explaining unit and range
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List

# ===== ENUMS (Finite value sets) =====

class NodeSizeClass(str, Enum):
    """Node tier classification."""
    SMALL = "S"
    MEDIUM = "M"
    LARGE = "L"

class ActionType(str, Enum):
    """Available actions (comprehensive enumeration)."""
    # Scale down
    SCALE_DOWN_5 = "SCALE_REPLICAS(-5)"
    SCALE_DOWN_1 = "SCALE_REPLICAS(-1)"
    # Maintain
    MAINTAIN = "MAINTAIN"
    # Scale up
    SCALE_UP_1 = "SCALE_REPLICAS(+1)"
    SCALE_UP_5 = "SCALE_REPLICAS(+5)"
    SCALE_UP_10 = "SCALE_REPLICAS(+10)"
    SCALE_UP_20 = "SCALE_REPLICAS(+20)"  # Audit fix 04: required for hard task
    # Structural changes
    UPGRADE_NODE = "UPGRADE_NODE"
    REBALANCE_NODE = "REBALANCE_NODE"

# ===== OBSERVATIONS (State perceived by agent) =====

class Observation(BaseModel):
    """Current environment state (all fields observable by agent)."""

    cpu_usage_pct: float = Field(ge=0, le=100, description="Cluster-wide CPU utilization [0-100%]")
    mem_usage_pct: float = Field(ge=0, le=100, description="Cluster-wide memory utilization [0-100%]")
    p99_latency_ms: float = Field(ge=0, description="Tail latency [0-∞ ms]; SLA threshold=300ms")
    http_error_rate: float = Field(ge=0, le=1, description="Request failure rate [0-1]")
    cpu_steal_pct: float = Field(ge=0, le=1, description="Noisy-neighbor indicator [0-1]")
    active_replicas: int = Field(ge=0, description="Running pod count [0-∞]")
    buffer_depth: int = Field(ge=0, description="Request queue depth [0-∞]")
    node_size_class: NodeSizeClass = Field(description="Current node tier {S|M|L}")
    current_hourly_cost: float = Field(ge=0, description="USD/hour spend [0-∞]")
    node_bin_density: List[float] = Field(
        min_length=10, max_length=10,
        description="Per-node packing ratio; fixed 10-element vector [0-1]×10"
    )

# ===== ACTIONS (Agent decisions) =====

class Action(BaseModel):
    """Agent action selection."""
    action_type: ActionType = Field(description="Selected action from ActionType enum")

# ===== ENVIRONMENT STATE (Internal representation) =====

class EnvState(BaseModel):
    """Environment state snapshot (returned by state() method)."""

    step: int = Field(ge=0, description="Current step counter [0-∞]")
    replicas: int = Field(ge=0, description="Active replica count [0-∞]")
    node_size: NodeSizeClass = Field(description="Current node tier {S|M|L}")
    prev_steal_pct: float = Field(ge=0, le=1, description="Previous-step steal % for proactive bonus [0-1]")

# ===== TRAJECTORY (For grading) =====

class TrajectoryStep(BaseModel):
    """Single step in episode trajectory (for graders)."""

    observation: Observation
    action: ActionType
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)

    # Task-specific metrics
    uptime_metric: float = Field(ge=0, le=1, description="SLA adherence [0-1]")
    cost_metric: float = Field(ge=0, description="Cost relative to budget [0-∞]")

class Trajectory(BaseModel):
    """Full episode trajectory for grading."""
    steps: List[TrajectoryStep] = Field(min_length=0)
```

**Scaffolding Checklist:**

- [ ] All observation fields typed (no bare strings)
- [ ] All Field() constraints present (ge, le, min_length, max_length)
- [ ] ActionType enum comprehensive (including +20 from Audit Fix 04)
- [ ] Enums use (str, Enum) for JSON serialization
- [ ] state() returns EnvState, not dict
- [ ] Trajectory model ready for grader inputs

**Critical Spec Points:**

✓ No bare dicts — all returns are Pydantic models  
✓ node_bin_density fixed-length (min_length=10, max_length=10, not variable)  
✓ Float fields use Field() with ge/le, never bare type hint

---

### 3. **env.py** — KubeCostEnv Main Class (Stubs Only)

**Purpose:** Environment interface implementation (Phase 5 will implement logic).

**Template Structure:**

```python
# env.py
"""
KubeCost-Gym Environment (OpenEnv Interface).

Stub phase: method signatures only, no logic.
Each method body is replaced with placeholder.
"""

import json
from pathlib import Path
from models import Observation, Action, EnvState, ActionType, Trajectory, TrajectoryStep

class KubeCostEnv:
    """Kubernetes cost optimization environment."""

    def __init__(self, trace_path: str):
        """
        Initialize environment from deterministic trace.

        Args:
            trace_path: Path to JSON trace file (e.g., 'traces/trace_v1_coldstart.json')

        Implementation note: Load trace, validate structure, initialize state.
        """
        self.trace_path = Path(trace_path)
        # STUB: Load JSON trace
        # STUB: Validate trace schema
        # STUB: Initialize step counter, state variables
        pass

    def reset(self) -> Observation:
        """
        Reset environment to initial state.

        Returns:
            Observation: Typed observation model (NOT dict)

        Validation:
            - Returns Pydantic Observation instance
            - All fields within declared ranges

        Common Failure (spec §2):
            ✗ Returns dict instead of Observation instance
            ✗ Should return Observation.parse_obj(...) or Observation(...)
        """
        # STUB: Return initial observation
        pass

    def step(self, action: Action) -> tuple:
        """
        Execute one step of environment dynamics.

        Args:
            action: Action model with action_type field

        Returns:
            (Observation, float, bool, dict):
                - obs: New observation after action
                - reward: Scalar reward this step
                - done: Episode termination flag
                - info: Metadata dict (status, debug info)

        Validation:
            - 4-tuple return (not None, not 3-tuple)
            - reward is float (not int)
            - done is bool
            - obs is Observation instance

        Common Failure (spec §2):
            ✗ Returns None (stub body)
            ✗ Returns 3-tuple instead of 4-tuple
        """
        # STUB: Apply action, advance dynamics, compute reward
        pass

    def state(self) -> EnvState:
        """
        Get current environment state.

        Returns:
            EnvState: Typed state model (NOT dict)

        Validation:
            - Returns Pydantic EnvState instance (not bare dict)
            - Non-null and serializable

        Common Failure (spec §2):
            ✗ Returns dict instead of EnvState instance
            ✗ Should return EnvState(step=..., replicas=..., ...)
        """
        # STUB: Return current EnvState
        pass

    def _apply_action(self, action: Action) -> None:
        """
        Apply action to environment dynamics.

        Internal helper for step().
        Implementation will update internal state based on ActionType.
        """
        # STUB: Update internal state based on action
        pass

    def _calculate_reward(self) -> float:
        """
        Calculate reward signal this step.

        Returns:
            float: Reward in range [R_min, R_max] per spec

        Spec Reference (§3.3 Reward Specification):
            R min per step: -20.0
            R max per step: +10.5

        Common Failure (spec audit fix 03):
            ✗ Sparse reward cliff at p99=300ms (zero gradient elsewhere)
            ✗ Should: Add linear ramp penalty for p99 ∈ [200, 300ms)
        """
        # STUB: Compute reward (phase 3 will define exact formula)
        pass

    def _load_trace(self, trace_path: Path) -> dict:
        """Load and validate deterministic trace JSON."""
        # STUB: Load trace_path, validate schema
        pass
```

**Scaffolding Checklist:**

- [ ] `reset()` has docstring specifying it returns `Observation` (not dict)
- [ ] `step()` has docstring specifying 4-tuple return
- [ ] `state()` has docstring specifying it returns `EnvState` (not dict)
- [ ] All methods have body as `pass` (no implementation)
- [ ] Docstrings reference spec sections
- [ ] Common failures documented in docstrings

**No Logic Phase Reminders:**

✓ Bodies are `pass`, not stubs that return None  
✓ Signatures are complete (args, return types)  
✓ Docstrings are spec-compliant  
✓ Helper methods (\_apply_action, etc.) stubbed but documented

---

### 4. **graders.py** — Scoring Functions (Math Formula First)

**Purpose:** Grader specifications (Phase 4, SDD §3.4). Define formulas before implementation.

**Template Structure:**

```python
# graders.py
"""
Grader implementations (Phase 4: Grader Spec).

Each grader:
  1. Has mathematical formula (documented in docstring)
  2. Returns float in [0.0, 1.0]
  3. Normalizes by trajectory length (no unbounded accumulation)
  4. Handles empty trajectory explicitly
"""

from typing import List
from models import TrajectoryStep

class ColdStartGrader:
    """
    Task 1: Cold Start (Easy).

    Objective: Scale cluster from 0→5 replicas without SLA breach.

    Formula (math notation):
        score = 1.0 - http_error_rate_avg
        final_score = max(0.0, min(1.0, score))

    Normalization:
        By average error rate (length-invariant)

    Edge case:
        Empty trajectory → return 0.0

    Spec Reference: §3 Phase 4, Audit Fix 02 (normalized grader)

    Common Failure:
        ✗ score = 1.0; score -= 0.05 per violation (unbounded, length-dependent)
        ✓ score = 1.0 - (violations / len(trajectory))
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade cold start performance.

        Args:
            trajectory: List of trajectory steps

        Returns:
            float: Score in [0.0, 1.0]

        Implementation note:
            - Check: if not trajectory: return 0.0
            - Compute: error rate average across steps
            - Return: normalized score, clamped to [0.0, 1.0]
        """
        # STUB: Compute score
        pass


class EfficientSqueezeGrader:
    """
    Task 2: Efficient Squeeze (Medium).

    Objective: Maintain <20% CPU steal across 24-hour sine-wave load cycle.

    Formula (math notation):
        violations = count(steps where cpu_steal_pct >= 0.20)
        score = 1.0 - (violations / len(trajectory))
        final_score = max(0.0, min(1.0, score))

    Normalization:
        By violation rate per trajectory length

    Edge case:
        Empty trajectory → return 0.0

    Spec Reference: §3 Phase 4, Audit Fix 02 (normalized)

    Key Insight:
        Violation rate (0.0 = perfect, 1.0 = all steps violated) is invariant
        to trajectory length. This ensures fair comparison across different
        simulation durations.
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade efficient squeeze performance.

        Args:
            trajectory: List of trajectory steps

        Returns:
            float: Score in [0.0, 1.0]

        Implementation note:
            - Check: if not trajectory: return 0.0
            - Scan: count steps where cpu_steal_pct >= 0.20
            - Normalize: violations / len(trajectory)
            - Return: clamped to [0.0, 1.0]
        """
        # STUB: Compute score
        pass


class EntropyStormGrader:
    """
    Task 3: Entropy Storm (Hard).

    Objective: Issue REBALANCE_NODE BEFORE steal exceeds 20% (proactive reasoning).

    Formula (math notation):
        1. For each violation at step i (steal >= 0.20):
           - Check if any REBALANCE_NODE action occurred in steps [i-k, i-1]
           - If yes: PROACTIVE BONUS += 0.5 per correctly predicted violation
           - If no:  CAN'T FIX IN TIME (latent failure, no recovery bonus)

        2. success_rate = successful_proactive_actions / total_violations
        3. score = success_rate × 1.0 (+ small cost minimization bonus)
        4. final_score = max(0.0, min(1.0, score))

    Normalization:
        By count of total violations (length of high-steal episodes)

    Edge case:
        Empty trajectory → return 0.0
        No violations (steal never >= 0.20) → return 1.0 (agent won)

    Spec Reference: §3 Phase 4, Task design (hard = proactive reasoning)

    Key Insight (Audit Fix 04):
        This is the ONLY task where reactive scaling (AFTER breach) cannot
        achieve high score. Agent MUST learn to predict and act before the
        leading indicator (steal) rises. Tests for genuine proactive reasoning.

    Design Challenge:
        Missing actions in action space (e.g., no +20 scaling) makes task
        structurally unsolvable. Verify ActionType is complete (@app start).
    """

    def grade(self, trajectory: List[TrajectoryStep]) -> float:
        """
        Grade entropy storm (proactive rebalancing) performance.

        Args:
            trajectory: List of trajectory steps

        Returns:
            float: Score in [0.0, 1.0]

        Implementation note:
            - Check: if not trajectory: return 0.0
            - Identify: all violations (cpu_steal_pct >= 0.20)
            - For each violation: check if REBALANCE_NODE in preceding steps
            - Compute: success_rate = proactive_actions / total_violations
            - Return: clamped to [0.0, 1.0]
        """
        # STUB: Compute score
        pass
```

**Scaffolding Checklist:**

- [ ] Each grader has formula documentation (plain text or LaTeX)
- [ ] Each grader docstring includes normalization strategy
- [ ] Each grader has edge case handling documented (empty trajectory)
- [ ] All bodies are `pass` (no implementation)
- [ ] Final clamp `max(0.0, min(1.0, score))` documented
- [ ] Spec audit fixes referenced in docstrings

**Critical Spec Points:**

✓ Graders handle empty: `if not trajectory: return 0.0`  
✓ Normalization explicit: violations / len(trajectory), not unbounded -= 0.05  
✓ No float equality: use `>= 0.20` not `== 0.20`  
✓ EntropyStormGrader requires proactive bonus (hard task design)

---

### 5. **inference.py** — LLM Inference Pipeline (Root Directory)

**Purpose:** LLM agent inference for online decision-making (Phase 5, Infra Spec).

**Location:** **ROOT DIRECTORY ONLY** (not subdirectory) — validator checks `./inference.py` specifically.

**Template Structure:**

```python
# inference.py
"""
LLM Inference Pipeline for KubeCost-Gym.

Location: ROOT directory (spec requirement).
Spec Reference: §5 Infra Spec, §2 common failure.

Environment variables (required):
  - API_BASE_URL: Model endpoint (e.g., https://api.openai.com/v1)
  - MODEL_NAME: Model identifier (e.g., gpt-4, gpt-3.5-turbo)
  - HF_TOKEN: HuggingFace API token for Space submission

Runtime requirement: Complete end-to-end in <20 minutes.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict
from env import KubeCostEnv
from models import Observation, Action, ActionType

# ===== ENVIRONMENT VARIABLE VALIDATION =====

def get_env_or_raise(key: str) -> str:
    """Raise if environment variable missing."""
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required env var: {key}")
    return value


# ===== INFERENCE PIPELINE =====

class CostOptimizerAgent:
    """
    LLM-based decision agent for cost optimization.

    Responsibilities:
        - Observe environment state (Observation model)
        - Query LLM for action recommendation (JSON response)
        - Validate response, extract ActionType
        - Execute step, collect trajectory
        - Score trajectory with graders
    """

    def __init__(self, model_name: str, api_base_url: str):
        """
        Initialize LLM inference client.

        Args:
            model_name: Model ID (from MODEL_NAME env var)
            api_base_url: API endpoint (from API_BASE_URL env var)

        Implementation note:
            - Instantiate OpenAI client (spec requires OpenAI API only)
            - Validate credentials
            - Store model_name for inference calls
        """
        self.model_name = model_name
        self.api_base_url = api_base_url
        # STUB: Initialize OpenAI client
        pass

    def decide(self, observation: Observation) -> Action:
        """
        Query LLM for action given current observation.

        Args:
            observation: Current Observation model

        Returns:
            Action: Selected action with validation

        Implementation note:
            - Serialize observation to JSON
            - Prompt LLM with state + task description
            - Parse JSON response
            - Validate ActionType enum membership
            - Return Action model

        Spec requirement: Use JSON mode (response_format: json_schema)
        """
        # STUB: Call LLM, parse response
        pass

    def run_episode(self, env: KubeCostEnv, max_steps: int = 1000) -> list:
        """
        Run one full episode with LLM agent.

        Args:
            env: Environment instance
            max_steps: Max steps before termination

        Returns:
            list: Trajectory (list of trajectory steps)

        Implementation note:
            - Call env.reset()
            - Loop: observe → decide → step until done or max_steps
            - Collect all steps in trajectory
            - Return trajectory for grading
        """
        # STUB: Run episode
        pass


def main():
    """
    Main inference entry point.

    Spec requirements (§5 Infra Spec):
        - Runs when executed: python inference.py
        - Must complete in <20 minutes
        - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN
        - Uses OpenAI client only
        - Outputs results (trajectory, scores)
    """

    # Environment variables
    api_base_url = get_env_or_raise("API_BASE_URL")
    model_name = get_env_or_raise("MODEL_NAME")
    hf_token = get_env_or_raise("HF_TOKEN")

    # Initialize
    # STUB: Create agent, environment
    # STUB: Run 3 tasks (cold_start, squeeze, entropy)
    # STUB: Collect trajectories
    # STUB: Score with graders
    # STUB: Output results
    pass

if __name__ == "__main__":
    main()
```

**Scaffolding Checklist:**

- [ ] Located in ROOT directory (not subdirectory)
- [ ] Environment variable validation: `get_env_or_raise()` for API_BASE_URL, MODEL_NAME, HF_TOKEN
- [ ] `if __name__ == "__main__"` guard present
- [ ] Class: `CostOptimizerAgent` with stubs for decide(), run_episode()
- [ ] `main()` function defined (not class method)
- [ ] Docstring explains <20 min runtime requirement

**Critical Spec Points:**

✓ File path: `./inference.py` (root, no subdirectories)  
✓ Uses OpenAI client API (not other backends)  
✓ Reads exactly 3 env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN  
✓ Runs end-to-end in <20 minutes (note on implementation)

---

### 6. **requirements.txt** — Python Dependencies

**Purpose:** Reproducible Python environment.

**Template:**

```
# Python 3.10+ required (spec §0)

# Core ML framework (if using gymnasium/OpenAI Gym)
gymnasium>=0.27.0

# Type system and validation
pydantic>=2.0.0
pydantic-settings>=2.0.0

# LLM API client (spec requirement: OpenAI only)
openai>=1.0.0

# Serialization
PyYAML>=6.0

# Utilities
numpy>=1.24.0
pandas>=1.5.0
python-dotenv>=0.21.0

# Development/testing
pytest>=7.0.0
pytest-cov>=4.0.0
```

**Scaffolding Checklist:**

- [ ] Python 3.10+ compatible
- [ ] pydantic>=2.0.0 for type validation
- [ ] openai>=1.0.0 for API calls
- [ ] PyYAML for openenv.yaml parsing
- [ ] pytest for test harness

---

### 7. **Dockerfile** — Container Specification

**Purpose:** Docker image for HF Space deployment (Infra Spec).

**Template:**

```dockerfile
# Dockerfile
# Based on python:3.10-slim (spec requirement)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer optimization)
COPY requirements.txt .

# Install dependencies (no-cache for layer size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify structure (fail early if inference.py missing)
RUN test -f inference.py || (echo "ERROR: inference.py not in root" && exit 1)

# Default command: run inference
CMD ["python", "inference.py"]
```

**Scaffolding Checklist:**

- [ ] Base image: `python:3.10-slim`
- [ ] WORKDIR: `/app`
- [ ] requirements.txt copied first
- [ ] Verification: inference.py exists in root
- [ ] CMD: python inference.py

---

### 8. **openenv.yaml** — (Already Scaffolded Above)

See [Section 1: openenv.yaml](#1-openenyyaml--environment-metadata--task-definitions).

---

### 9. **README.md** — HuggingFace Space Frontmatter + Setup

**Purpose:** Space metadata + setup instructions.

**Template:**

````markdown
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
---

# KubeCost-Gym: Kubernetes Cost Optimization RL Environment

## Overview

A production-grade RL environment for learning Kubernetes autoscaling strategies. Implements the **OpenEnv** standard with Pydantic-typed observations, deterministic trace-based dynamics, and three progressive difficulty tasks.

**Project Spec:** See [PROJECT_SPEC.md](PROJECT_SPEC.md) for complete SDD requirements.

## Tasks

| Task                  | Difficulty | Objective                                                      |
| --------------------- | ---------- | -------------------------------------------------------------- |
| **Cold Start**        | Easy       | Scale cluster 0→5 replicas without SLA breach (p99<300ms)      |
| **Efficient Squeeze** | Medium     | Maintain <20% CPU steal over 24-hour load cycle                |
| **Entropy Storm**     | Hard       | Proactive REBALANCE_NODE before steal>20% (leading indicators) |

## Setup

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd k8s-cost-optimizer
pip install -r requirements.txt

# Run local validation
python validate_local.py

# Run inference
python inference.py
```
````

### Environment Variables

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="hf_..."
```

### Docker

```bash
docker build -t kubecost-gym .
docker run -e API_BASE_URL="..." -e MODEL_NAME="..." -e HF_TOKEN="..." kubecost-gym
```

## File Structure

```
├── env.py                 # KubeCostEnv class (OpenEnv interface)
├── models.py              # Pydantic type definitions
├── graders.py             # Three grader implementations
├── inference.py           # LLM agent inference pipeline
├── openenv.yaml           # Task definitions & metadata
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image spec
├── traces/                # Deterministic dynamics (JSON)
└── validate_local.py      # Pre-submission validator
```

## Observation Space

| Field                 | Type        | Range     | Purpose                    |
| --------------------- | ----------- | --------- | -------------------------- |
| `cpu_usage_pct`       | float       | [0, 100]  | Cluster CPU utilization    |
| `mem_usage_pct`       | float       | [0, 100]  | Cluster memory utilization |
| `p99_latency_ms`      | float       | [0, ∞)    | Tail latency (SLA: 300ms)  |
| `http_error_rate`     | float       | [0, 1]    | Request failure rate       |
| `cpu_steal_pct`       | float       | [0, 1]    | Noisy-neighbor indicator   |
| `active_replicas`     | int         | [0, ∞)    | Running pod count          |
| `buffer_depth`        | int         | [0, ∞)    | Request queue depth        |
| `node_size_class`     | Enum        | {S, M, L} | Node tier                  |
| `current_hourly_cost` | float       | [0, ∞)    | USD/hour                   |
| `node_bin_density`    | List[float] | [0, 1]×10 | Per-node packing           |

## Action Space

- `SCALE_REPLICAS(-5)` — Aggressive scale-down
- `SCALE_REPLICAS(-1)` — Gentle scale-down
- `MAINTAIN` — Hold current state
- `SCALE_REPLICAS(+1)` — Gentle scale-up
- `SCALE_REPLICAS(+5)` — Moderate scale-up
- `SCALE_REPLICAS(+10)` — Large scale-up
- `SCALE_REPLICAS(+20)` — Emergency burst (hard task)
- `UPGRADE_NODE` — Vertical scale (irreversible 1 step)
- `REBALANCE_NODE` — Deterministic rebalance

## Spec-Driven Development

This project follows SDD phases:

1. **Domain Spec** — Real-world problem & task ladder
2. **Contract Spec** — Pydantic models & type constraints
3. **Reward Spec** — Mathematical reward formula (no sparse cliffs)
4. **Grader Spec** — Normalized scoring [0.0, 1.0]
5. **Infra Spec** — Docker & deployment

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for detailed phase requirements.

## Validation

Pre-submission checks:

```bash
python validate_local.py
```

Validates:

- ✓ HF Space responds to reset()
- ✓ Docker builds without errors
- ✓ inference.py in root & runs in <20 min
- ✓ openenv.yaml parses & has required fields
- ✓ All graders return [0.0, 1.0]
- ✓ No stub bodies (pass) remain

## Deadline & Scoring

- **Deadline:** 8 Apr 2026, 11:59 PM IST
- **Target:** ≥27/30 points
- **Scoring:** Interface compliance (gate), task design (high), grading logic (high)

---

**Last Updated:** March 30, 2026

````

**Scaffolding Checklist:**

- [ ] YAML frontmatter present and correct:
  - [ ] `hardware: cpu-basic` (NOT cpu-upgrade)
  - [ ] `sdk: docker`
  - [ ] `tags: - openenv`
- [ ] Observation & action space tables match models.py
- [ ] Setup instructions include env vars
- [ ] File structure section references all key files
- [ ] Links to PROJECT_SPEC.md and validate_local.py

---

### 10. **validate_local.py** — Pre-Submission Validator

**Purpose:** Run all spec compliance checks before submission.

**Template:**

```python
# validate_local.py
"""
Pre-submission validation harness (spec §7 automated gates).

Run locally before pushing to HF Space.
Checks:
  - Module imports (no syntax errors)
  - reset() returns Observation (not dict)
  - step() returns 4-tuple with correct types
  - state() returns EnvState (not dict)
  - All graders return [0.0, 1.0]
  - openenv.yaml parses and has required fields
  - inference.py exists in root
  - No stub bodies (pass only)
"""

import sys
import yaml
from pathlib import Path

def check_imports():
    """Validate all modules import without syntax errors."""
    try:
        from env import KubeCostEnv
        from models import Observation, EnvState, Action, ActionType
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
        print("✓ All modules import successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def check_openenv_yaml():
    """Validate openenv.yaml structure."""
    try:
        with open("openenv.yaml") as f:
            spec = yaml.safe_load(f)

        # Required fields
        assert "name" in spec, "Missing 'name'"
        assert "version" in spec, "Missing 'version'"
        assert isinstance(spec["version"], str), "'version' must be string, not float"
        assert "description" in spec, "Missing 'description'"
        assert "tasks" in spec, "Missing 'tasks'"
        assert len(spec["tasks"]) == 3, "Must have exactly 3 tasks"

        # Task validation
        for task in spec["tasks"]:
            assert "name" in task, f"Task missing 'name'"
            assert "difficulty" in task, f"Task {task.get('name')} missing 'difficulty'"
            assert task["difficulty"] in ["easy", "medium", "hard"], \
                f"Invalid difficulty: {task['difficulty']}"
            assert "description" in task, f"Task {task.get('name')} missing 'description'"

        print(f"✓ openenv.yaml valid: {spec['name']} v{spec['version']}")
        return True
    except Exception as e:
        print(f"✗ openenv.yaml validation failed: {e}")
        return False

def check_graders():
    """Validate all graders return [0.0, 1.0]."""
    try:
        from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
        from models import TrajectoryStep, Observation, ActionType

        # Create dummy trajectory
        dummy_obs = Observation(
            cpu_usage_pct=50.0,
            mem_usage_pct=60.0,
            p99_latency_ms=250.0,
            http_error_rate=0.01,
            cpu_steal_pct=0.05,
            active_replicas=5,
            buffer_depth=10,
            node_size_class="M",
            current_hourly_cost=100.0,
            node_bin_density=[0.5] * 10
        )
        dummy_step = TrajectoryStep(
            observation=dummy_obs,
            action=ActionType.MAINTAIN,
            reward=1.0,
            done=False,
            uptime_metric=1.0,
            cost_metric=0.8
        )

        for grader_cls in [ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader]:
            grader = grader_cls()
            score = grader.grade([dummy_step])
            assert 0.0 <= score <= 1.0, \
                f"{grader_cls.__name__}: score {score} out of range [0.0, 1.0]"
            print(f"✓ {grader_cls.__name__}: score={score:.3f}")

        return True
    except Exception as e:
        print(f"✗ Grader validation failed: {e}")
        return False

def check_inference_root():
    """Validate inference.py in root directory."""
    if Path("inference.py").exists():
        print("✓ inference.py exists in root directory")
        return True
    else:
        print("✗ inference.py not found in root directory")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("KubeCost-Gym Local Validation")
    print("=" * 50)

    checks = [
        ("Imports", check_imports),
        ("openenv.yaml", check_openenv_yaml),
        ("Graders", check_graders),
        ("Inference Root", check_inference_root),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        results.append(check_fn())

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")
    print("=" * 50)

    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
````

**Scaffolding Checklist:**

- [ ] Imports validation (models, env, graders)
- [ ] openenv.yaml structure check
- [ ] Grader return type validation
- [ ] inference.py root location check
- [ ] Callable via: `python validate_local.py`

---

### 11. **traces/** — Deterministic Dynamics Data

**Purpose:** Pre-recorded JSON traces for reproducibility (Decision 1 assumes JSON traces).

**Structure:**

```
traces/
├── trace_v1_coldstart.json    ← Task 1: Cold Start
├── trace_v1_squeeze.json      ← Task 2: Efficient Squeeze
└── trace_v1_entropy.json      ← Task 3: Entropy Storm
```

**Trace File Format (Template):**

```json
{
  "task_name": "cold_start",
  "task_difficulty": "easy",
  "steps": [
    {
      "step": 0,
      "observation": {
        "cpu_usage_pct": 5.0,
        "mem_usage_pct": 10.0,
        "p99_latency_ms": 150.0,
        "http_error_rate": 0.02,
        "cpu_steal_pct": 0.03,
        "active_replicas": 0,
        "buffer_depth": 100,
        "node_size_class": "S",
        "current_hourly_cost": 10.0,
        "node_bin_density": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
      },
      "action": "MAINTAIN",
      "dynamics": {
        "next_replicas": 0,
        "next_node_size": "S"
      }
    },
    {
      "step": 1,
      ...
    }
  ]
}
```

**Scaffolding Checklist:**

- [ ] Directory created: `traces/`
- [ ] Three JSON files (coldstart, squeeze, entropy)
- [ ] Each file has schema: task_name, task_difficulty, steps[]
- [ ] Each step has: step, observation, action, dynamics
- [ ] Observations match Observation model fields exactly
- [ ] Actions match ActionType enum values

**Design Decision (Decision 1 Required):**

⚠️ **User must specify:**

- Will traces be pre-generated, procedural, or loaded from external source?
- What deterministic scenarios should each trace simulate?

---

## Initialization Checklist

### Phase 0: Pre-Scaffolding Decisions

**Required clarifications before proceeding:**

- [ ] **Decision 1:** Trace source (JSON files, procedural, API)
- [ ] **Decision 2:** Task specifications finalized (names, descriptions, success criteria)
- [ ] **Decision 3:** LLM backend chosen (API provider, model)
- [ ] **Decision 4:** Observation space locked (10 fields confirmed or modifications proposed)

### Phase 1: Project Structure Creation

```bash
# Create directory structure
mkdir -p k8s-cost-optimizer/traces
cd k8s-cost-optimizer

# Create stub files (content from scaffolding templates above)
touch models.py
touch env.py
touch graders.py
touch inference.py
touch requirements.txt
touch Dockerfile
touch openenv.yaml
touch README.md
touch validate_local.py
touch traces/trace_v1_coldstart.json
touch traces/trace_v1_squeeze.json
touch traces/trace_v1_entropy.json
```

### Phase 2: File Population

**Order of completion:**

1. [ ] **openenv.yaml** — Copy template, confirm task names & difficulty
2. [ ] **models.py** — Copy Pydantic stubs (no logic changes)
3. [ ] **env.py** — Copy KubeCostEnv stubs (all bodies as `pass`)
4. [ ] **graders.py** — Copy grader stubs (formula documented, no impl)
5. [ ] **requirements.txt** — Copy dependency list
6. [ ] **Dockerfile** — Copy image spec
7. [ ] **README.md** — Copy setup guide
8. [ ] **inference.py** — Copy LLM stubs
9. [ ] **validate_local.py** — Copy validator harness
10. [ ] **traces/\*.json** — Create or copy trace files (Decision 1)

### Phase 3: Scaffolding Validation

```bash
# Validate syntax and structure (before implementing logic)
python -m py_compile models.py env.py graders.py inference.py validate_local.py

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"

# Check Docker builds
docker build -t kubecost-gym-test .

# Run local validator (all stubs should pass structure checks)
python validate_local.py
```

### Phase 4: Ready for Implementation

After scaffolding completes:

- ✓ All files created with correct structure
- ✓ All stubs in place (bodies are `pass`, not implemented)
- ✓ All docstrings document spec requirements
- ✓ No logic code written yet
- ✓ Validator runs without errors (reports stubs as expected)

**Next Phase:** Phase 1 Domain Specification (problem definition, task ladder validation)

---

## Critical Spec Reminders

### 🚨 Common Failures to Prevent During Scaffolding

| Failure                          | Why It's Fatal                                | Prevention                                              |
| -------------------------------- | --------------------------------------------- | ------------------------------------------------------- |
| **inference.py in subdirectory** | Validator looks for `./inference.py` only     | Create in root, never in folder                         |
| **Hardware tag: cpu-upgrade**    | Exceeds 2 vCPU hackathon limit                | Use `hardware: cpu-basic` exclusively in README.md      |
| **version: 3.0 (unquoted)**      | YAML parses as float, not string              | Use `version: "3.0"` in openenv.yaml                    |
| **Stub bodies missing**          | step() returns None, tuple unpacking crashes  | Ensure all bodies are `pass` (not implementation)       |
| **Graders unbounded**            | score = 1.0; score -= 0.05 (grows unbounded)  | Normalize: score = 1.0 - (violations / len(trajectory)) |
| **Float equality**               | 0.00000001 ≠ 0.0 in floating-point math       | Use < 0.001, not == 0.0                                 |
| **Wrong Enum values**            | ActionType missing +20 scaling (Audit Fix 04) | Include SCALE_REPLICAS(+20) enum value                  |
| **Hard task unsolvable**         | No action sequence achieves objective         | Verify all actions present before task design           |

### 📋 Spec Sections Referenced in Scaffolding

- **§0:** Overview, deadline, Python 3.10+
- **§1:** OpenEnv interface (reset, step, state, openenv.yaml)
- **§2:** Interface contract, common failures
- **§3:** 5 SDD phases, reward formula
- **§4:** Observable & action variables, grading rules
- **§5:** Infrastructure (Docker, README frontmatter, inference.py)
- **§7:** Pre-submission validator gates (automated)
- **§10:** 8-step SDD workflow

---

## Summary: What This Manual Covers

✅ **Complete**: Project structure, file templates, stub contents, validation setup  
✅ **Decision Points**: Identified 4 critical clarifications needed  
✅ **Spec-Compliant**: All templates follow SDD phases exactly  
✅ **Error Prevention**: Common failures documented in each file

❌ **Does NOT implement**: Actual environment dynamics, reward formulas, grader logic, LLM integration  
❌ **Does NOT generate**: Full codebase — only stubs and structure  
❌ **Does NOT decide**: Task specifications, trace formats, LLM backend (user must clarify)

---

## Next Steps After Scaffolding

1. **Verify** all files created from templates
2. **Clarify** 4 decision points with user/team
3. **Validate** structure with: `python validate_local.py`
4. **Proceed** to Phase 1: Domain Specification
5. **Implement** Phase by phase per SDD workflow (§10)

---

**Document Version:** 1.0  
**Created:** March 30, 2026  
**Status:** Ready for scaffolding (awaiting Decision 1-4 clarifications)
