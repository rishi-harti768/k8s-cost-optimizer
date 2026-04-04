# models.py
"""
Pydantic type definitions for KubeCost-Gym (Phase 2: Contract Spec).

All fields include:
  - Type annotations (no bare strings)
  - Field() constraints (min/max, enums, list bounds)
  - Docstrings explaining unit and range

Reference: PROJECT_SPEC.md §3 Phase 2 Contract Specification
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, confloat, conlist

__all__ = [
    "NodeSizeClass",
    "ActionType",
    "Observation",
    "Action",
    "EnvState",
    "TrajectoryStep",
    "Trajectory",
    "TraceStep",
    "TraceData",
]


# ===== ENUMS (Finite value sets) =====

class NodeSizeClass(str, Enum):
    """Node tier classification."""
    SMALL = "S"
    MEDIUM = "M"
    LARGE = "L"


class ActionType(str, Enum):
    """Available actions (comprehensive enumeration).
    
    Includes SCALE_REPLICAS(+20) per Audit Fix 04 requirement.
    Hard task requires this for emergency burst absorption.
    """
    # Scale down
    SCALE_DOWN_5 = "SCALE_REPLICAS(-5)"
    SCALE_DOWN_1 = "SCALE_REPLICAS(-1)"
    # Maintain
    MAINTAIN = "MAINTAIN"
    # Scale up
    SCALE_UP_1 = "SCALE_REPLICAS(+1)"
    SCALE_UP_5 = "SCALE_REPLICAS(+5)"
    SCALE_UP_10 = "SCALE_REPLICAS(+10)"
    SCALE_UP_20 = "SCALE_REPLICAS(+20)"  # Audit Fix 04: CRITICAL for hard task solvability
    # Structural changes
    UPGRADE_NODE = "UPGRADE_NODE"
    REBALANCE_NODE = "REBALANCE_NODE"


# ===== OBSERVATIONS (State perceived by agent) =====

class Observation(BaseModel):
    """Current environment state (all fields observable by agent).
    
    Reference: PROJECT_SPEC.md §4 Observation Space
    """
    
    cpu_usage_pct: float = Field(
        ge=0, le=100,
        description="Cluster-wide CPU utilization [0-100%]"
    )
    mem_usage_pct: float = Field(
        ge=0, le=100,
        description="Cluster-wide memory utilization [0-100%]"
    )
    p99_latency_ms: float = Field(
        ge=0,
        description="Tail latency [0-∞ ms]; SLA threshold=300ms"
    )
    http_error_rate: float = Field(
        ge=0, le=1,
        description="Request failure rate [0-1]"
    )
    cpu_steal_pct: float = Field(
        ge=0, le=1,
        description="Noisy-neighbor indicator [0-1]; critical for Task 3"
    )
    active_replicas: int = Field(
        ge=0,
        description="Running pod count [0-∞]"
    )
    buffer_depth: int = Field(
        ge=0,
        description="Request queue depth [0-∞]"
    )
    node_size_class: NodeSizeClass = Field(
        description="Current node tier {S|M|L}"
    )
    current_hourly_cost: float = Field(
        ge=0,
        description="USD/hour spend [0-∞]"
    )
    node_bin_density: conlist(confloat(ge=0.0, le=1.0), min_length=10, max_length=10) = Field(
        description="Per-node packing ratio; fixed 10-element vector [0-1]×10"
    )

    class Config:
        """Use enum values (not names) in JSON serialization."""
        use_enum_values = True


# ===== ACTIONS (Agent decisions) =====

class Action(BaseModel):
    """Agent action selection."""
    action_type: ActionType = Field(description="Selected action from ActionType enum")

    class Config:
        """Allow enum values."""
        use_enum_values = False


# ===== ENVIRONMENT STATE (Internal representation) =====

class EnvState(BaseModel):
    """Environment state snapshot (returned by state() method).
    
    CRITICAL: Returned by state() method, NOT dict (spec §2).
    """
    
    step: int = Field(
        ge=0,
        description="Current step counter [0-∞]"
    )
    replicas: int = Field(
        ge=0,
        description="Active replica count [0-∞]"
    )
    node_size: NodeSizeClass = Field(
        description="Current node tier {S|M|L}"
    )
    prev_steal_pct: float = Field(
        ge=0, le=1,
        description="Previous-step steal % for proactive bonus calculation [0-1]"
    )

    class Config:
        """Use enum values in JSON serialization."""
        use_enum_values = True


# ===== TRAJECTORY (For grading) =====

class TrajectoryStep(BaseModel):
    """Single step in episode trajectory (for graders).
    
    Reference: PROJECT_SPEC.md §4 Grader Specification
    
    Note: Remove redundant metrics (uptime_metric, cost_metric) — graders compute these from observation.
    """
    
    observation: Observation
    action: ActionType
    reward: float = Field(description="Reward this step")
    done: bool = Field(description="Episode termination flag")
    info: dict = Field(
        default_factory=dict,
        description="Metadata dict"
    )

    class Config:
        """Use enum values in JSON serialization."""
        use_enum_values = True


class Trajectory(BaseModel):
    """Full episode trajectory for grading.
    
    Used by all three graders (ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader).
    """
    steps: List[TrajectoryStep] = Field(
        min_length=0,
        description="List of trajectory steps"
    )


# ===== TRACE DATA (For loading from JSON files) =====

class TraceStep(BaseModel):
    """Single step record from a deterministic trace JSON file.
    
    Reference: PROJECT_SPEC.md §3 Phase 1 Determinism Guarantee
    """
    step: int = Field(description="Step index in trace")
    observation: Observation = Field(description="Observation at this step")
    dynamics: dict = Field(
        default_factory=dict,
        description="Optional dynamics metadata (reason, etc.)"
    )


class TraceData(BaseModel):
    """Full deterministic trace loaded from JSON file.
    
    Pydantic automatically validates all nested Observation fields, step numbers,
    and ensures the trace structure is well-formed.
    
    Reference: PROJECT_SPEC.md §3 Phase 1 Determinism Guarantee
    """
    task_name: str = Field(description="Task identifier")
    task_difficulty: str = Field(description="Task difficulty (easy|medium|hard)")
    steps: List[TraceStep] = Field(
        min_length=1,
        description="Non-empty list of trace steps"
    )
