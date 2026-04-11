---
title: k8s_cost_optimizer
emoji: 🚀
colorFrom: red
colorTo: orange
sdk: docker
hardware: cpu-basic
tags:
  - openenv
---

# KubeCost-Gym v3.1

KubeCost-Gym v3.1 simulates a Kubernetes cluster. It provides a deterministic reinforcement learning environment to train agents in proactive cost optimization. 

## Environment Dynamics

The simulation balances infrastructure cost against system reliability. The environment derives all dynamics from pre-recorded JSON traces to guarantee absolute determinism. 

The environment rewards successful outcomes, never specific actions. The reward formula balances service uptime against hourly cost. It applies a continuous linear penalty ramp for tail latencies between 200ms and 300ms. Agents earn proactive bonuses by stabilizing the cluster before latency thresholds breach.

## Setup Instructions

Ensure your system runs Python 3.10 or newer.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   - `API_BASE_URL`: Base URL for the LLM API.
   - `MODEL_NAME`: Name of the target model.
   - `HF_TOKEN`: HuggingFace authentication token.

3. **Run inference:**
   ```bash
   python inference.py
   ```

## Observation Space

Agents perceive the environment through a dimensionally fixed, typed observation space.

| Field | Type | Range | Purpose |
| --- | --- | --- | --- |
| `cpu_usage_pct` | Float | [0, 100] | Cluster-wide CPU utilization |
| `mem_usage_pct` | Float | [0, 100] | Cluster-wide memory utilization |
| `p99_latency_ms` | Float | [0, ∞) | Tail latency; SLA threshold equals 300ms |
| `http_error_rate` | Float | [0, 1] | Request failure rate |
| `cpu_steal_pct` | Float | [0, 1] | Noisy-neighbor indicator |
| `active_replicas` | Integer | [0, ∞) | Running pod count |
| `buffer_depth` | Integer | [0, ∞) | Request queue depth |
| `node_size_class` | Enum | {S, M, L} | Current node tier |
| `current_hourly_cost`| Float | [0, ∞) | USD/hour spend |
| `node_bin_density` | List[Float] | [0, 1] × 10 | Per-node packing; fixed 10-element vector |

## Action Space

Agents manage the cluster by issuing discrete scaling and balancing commands.

- **`SCALE_REPLICAS(-5)`**: Aggressive scale-down.
- **`SCALE_REPLICAS(-1)`**: Gentle scale-down.
- **`MAINTAIN`**: Hold current state.
- **`SCALE_REPLICAS(+1)`**: Gentle scale-up.
- **`SCALE_REPLICAS(+5)`**: Moderate scale-up.
- **`SCALE_REPLICAS(+10)`**: Large scale-up.
- **`SCALE_REPLICAS(+20)`**: Emergency burst absorption.
- **`UPGRADE_NODE`**: Vertical scale.
- **`REBALANCE_NODE`**: Deterministic rebalance.

## Core Tasks

The environment trains agents through three progressive tasks. Each task requires a unique decision-making strategy.

### Task 1: Cold Start (Easy)
Scale the cluster from zero replicas to five in minimum steps. Complete this operation without breaching the SLA threshold.

### Task 2: Efficient Squeeze (Medium)
Maintain sub-20% steal percentage across a 24-hour sinusoidal load cycle. Minimize infrastructure cost simultaneously.

### Task 3: Entropy Storm (Hard)
Anticipate noisy-neighbor issues using leading indicators. Issue `REBALANCE_NODE` before the steal percentage exceeds 20%.
