---
title: k8s_cost_optimizer
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: docker
hardware: cpu-basic
tags:
  - openenv
---

# KubeCost-Gym v3.1

KubeCost-Gym v3.1 simulates a Kubernetes cluster autoscaling environment for reinforcement learning agents. Agents must balance infrastructure costs against service reliability by managing replica counts and node sizes.

## Technical Specifications

The environment adheres to the OpenEnv protocol and runs within a constrained hardware profile (2 vCPU, 8 GB RAM).

### Observation Space

The agent receives a 10-dimensional observation at each step.

| Field | Range | Description |
|---|---|---|
| `cpu_usage_pct` | [0, 100] | Cluster CPU utilization (%) |
| `mem_usage_pct` | [0, 100] | Cluster memory utilization (%) |
| `p99_latency_ms` | [0, ∞) | Tail latency (SLA threshold: 300ms) |
| `http_error_rate` | [0, 1] | Fraction of failed requests |
| `cpu_steal_pct` | [0, 1] | CPU time lost to noisy neighbors |
| `active_replicas` | [0, 200] | Count of running pods |
| `buffer_depth` | [0, ∞) | Count of requests in queue |
| `node_size_class` | {S, M, L} | Current server tier |
| `current_hourly_cost` | [0, ∞) | Current USD spend per hour |
| `node_bin_density` | [0, 1] × 10 | Per-node resource packing ratios |

### Action Space

Agents select from 9 discrete actions to modify cluster capacity.

| Action | Result |
|---|---|
| `SCALE_REPLICAS(-5)` | Subtract 5 pods |
| `SCALE_REPLICAS(-1)` | Subtract 1 pod |
| `MAINTAIN` | No change |
| `SCALE_REPLICAS(+1)` | Add 1 pod |
| `SCALE_REPLICAS(+5)` | Add 5 pods |
| `SCALE_REPLICAS(+10)` | Add 10 pods |
| `SCALE_REPLICAS(+20)` | Add 20 pods (emergency burst) |
| `UPGRADE_NODE` | Increment node tier (S → M or M → L) |
| `REBALANCE_NODE` | Suppress steal for 3 steps |

## Scoring and Rewards

The environment evaluates agents based on three metrics: Uptime, Cost, and Proactivity.

### Reward Function
The per-step reward ($R$) follows a weighted formula:
$$R = (10.0 \times Uptime) - (5.0 \times \frac{Cost}{Budget}) - RampPenalty(p99) - SLABreach(p99) + ProactiveBonus$$

*   **Uptime**: +1.0 if $p99 < 300ms$, else 0.
*   **Ramp Penalty**: Linear penalty between 200ms and 300ms to provide a dense training signal.
*   **SLA Breach**: -20.0 penalty if $p99 \ge 300ms$.
*   **Proactive Bonus**: +0.5 if $p99 < 300ms$ and CPU steal decreased from the previous step.

### Task Scoring
Task graders normalize performance into a score between **0.1** and **0.9**. 
*   **0.1**: Failure, empty trajectory, or total SLA breach.
*   **0.9**: Optimal performance with zero SLA violations and minimal cost.

## Tasks

### 1. Cold Start (Easy)
Scale a cluster from 0 to 5 replicas. Initial error rates exceed 60%. The agent succeeds by bringing replicas online quickly enough to satisfy the 300ms SLA.

### 2. Efficient Squeeze (Medium)
Navigate a 24-hour sinusoidal load cycle. The agent must scale down during troughs to save cost and scale up before peak demand breaches the SLA.

### 3. Entropy Storm (Hard)
Defend against bursty noisy-neighbor interference. The agent must use leading indicators (latent steal signals) to issue `REBALANCE_NODE` before $cpu\_steal\_pct$ exceeds 20%.

## Development

### Setup
1. Install Python 3.10+.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

### Execution
*   **Generate Traces**: `python generate_traces.py`
*   **Run Inference**: `python inference.py`
*   **Validate Locally**: `python validate_local.py`

## Repository Structure

*   `server/`: Contains `K8sCostOptimizerEnvironment` and the FastAPI app.
*   `graders.py`: Implementation of the strict [0.1, 0.9] scoring logic.
*   `models.py`: Pydantic definitions for all observation and action types.
*   `traces/`: Deterministic workload data in JSON format.
