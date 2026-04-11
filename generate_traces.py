"""
generate_traces.py — Realistic K8s workload trace generator.

Each of the 15 traces is deterministically seeded but uses distinct
parameters and stochastic models per task × variant combination,
producing genuinely different workload dynamics.

Task physics:
  cold_start      — exponential demand ramp from 0 replicas; error driven
                    by capacity starvation; latency inversely correlated.
  efficient_squeeze — full-provisioned cluster on a 24h demand cycle;
                    steal is the dominant pressure; cost tracks replicas.
  entropy_storm   — noisy-neighbor cluster; steal spikes drive error and
                    latency; cascading effects modeled as carry-over state.

Run:
    python generate_traces.py
    TRACE_STEPS=25 TRACE_MAX_COUNT=15 python generate_traces.py
"""

import json
import math
import os
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
TRACE_STEPS = int(os.getenv("TRACE_STEPS", "25"))
TRACE_MAX_COUNT = int(os.getenv("TRACE_MAX_COUNT", "9999"))


# ── Helpers ───────────────────────────────────────────────────────────────────


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def gn(rng: random.Random, sigma: float) -> float:
    """Zero-mean Gaussian noise."""
    return rng.gauss(0.0, sigma)


def realistic_bin_density(rng: random.Random, avg: float, n: int = 10) -> list[float]:
    """
    Bimodal node-packing vector.

    Real clusters are heterogeneous: some nodes are packed tightly (hot),
    others are half-empty (cold). avg controls the mix point.
    """
    result = []
    for _ in range(n):
        if rng.random() < avg:
            # hot node: 60-95% packed
            v = clamp(rng.gauss(0.75, 0.10), 0.05, 0.99)
        else:
            # cold node: 10-50% packed
            v = clamp(rng.gauss(0.28, 0.10), 0.01, 0.60)
        result.append(round(v, 4))
    rng.shuffle(result)
    return result


# ── Cold-start traces ─────────────────────────────────────────────────────────
#
# Physics: service boots from 0 replicas while traffic ramps up.
# - error_rate starts at 0.6+ (no capacity) and falls as replicas spin up
# - latency is inversely correlated with available capacity
# - steal stays low on freshly-provisioned nodes
# - cost grows with each replica added
#
# Variant differences:
#   baseline    — standard 4-step scale cadence, error floor 0.04
#   gradual     — 6-step cadence, error slow to recover (floor 0.09)
#   aggressive  — 2-step cadence, node upgrades from S→M mid-trace
#   failed      — replicas barely scale; error never truly recovers (floor 0.30)
#   optimal     — 3-step cadence, early node upgrade, cleanest recovery

_COLD_START_VARIANTS = {
    # scale_every: add 1 replica every N steps
    # error_floor: lowest achievable error once fully provisioned
    # max_replicas: replica ceiling for this variant
    # upgrade_at:  step at which node upgrades S→M (None = never)
    # steal_base:  baseline steal for this variant
    "baseline":   dict(scale_every=4, error_floor=0.04, max_replicas=5, upgrade_at=None, steal_base=0.020),
    "gradual":    dict(scale_every=6, error_floor=0.09, max_replicas=5, upgrade_at=None, steal_base=0.015),
    "aggressive": dict(scale_every=2, error_floor=0.02, max_replicas=8, upgrade_at=12,   steal_base=0.030),
    "failed":     dict(scale_every=9, error_floor=0.30, max_replicas=2, upgrade_at=None, steal_base=0.045),
    "optimal":    dict(scale_every=3, error_floor=0.01, max_replicas=6, upgrade_at=8,    steal_base=0.018),
}


def generate_cold_start(variant: str, steps: int, rng: random.Random) -> list[dict]:
    p = _COLD_START_VARIANTS[variant]
    active_replicas = 0
    node_size = "S"
    steps_data = []

    for i in range(steps):
        t = i / max(steps - 1, 1)

        # CPU demand: exponential ramp — traffic builds from cold
        cpu = clamp(18.0 + 62.0 * (1.0 - math.exp(-t * 3.5)) + gn(rng, 2.5), 0.0, 100.0)

        # Memory: lags CPU by ~3 steps (JVM warmup, cache fill)
        t_mem = max(0.0, i - 3) / max(steps - 1, 1)
        mem = clamp(12.0 + 54.0 * (1.0 - math.exp(-t_mem * 2.5)) + gn(rng, 1.8), 0.0, 100.0)

        # Scale replicas on cadence (failed variant is sluggish)
        if variant == "failed":
            if i == 10:
                active_replicas = 1   # first replica arrives very late
            elif i == 20:
                active_replicas = min(2, p["max_replicas"])
        else:
            if i > 0 and i % p["scale_every"] == 0:
                active_replicas = min(active_replicas + 1, p["max_replicas"])

        # Node upgrade
        if p["upgrade_at"] and i == p["upgrade_at"]:
            node_size = "M"

        # Error rate: decays with capacity; floor is variant-specific
        capacity_ratio = active_replicas / max(1, p["max_replicas"])
        base_error = clamp(
            0.62 * math.exp(-capacity_ratio * 4.8) + p["error_floor"] + gn(rng, 0.018),
            0.0, 1.0,
        )

        # Steal: modest on fresh nodes; small sinusoidal jitter
        steal = clamp(
            p["steal_base"] + 0.015 * math.sin(i / 4.0) + gn(rng, 0.004),
            0.0, 0.15,
        )

        # Latency: high when under-capacity, drops as replicas land
        #   base ~350 ms at 0 capacity → ~130 ms when fully provisioned
        p99 = clamp(
            360.0 - 230.0 * capacity_ratio + 75.0 * base_error + gn(rng, 12.0),
            40.0, 600.0,
        )

        # Cost: node baseline + per-replica cost
        node_base = {"S": 10.0, "M": 20.0, "L": 40.0}[node_size]
        cost = round(clamp(node_base + active_replicas * 1.0 + gn(rng, 0.15), 0.0, 500.0), 2)

        # Buffer: inflates under capacity starvation
        buf = clamp(int(70 + base_error * 160 + rng.randint(-8, 8)), 0, 300)

        # Bin density: scales with provisioned capacity
        avg_density = clamp(0.20 + capacity_ratio * 0.50, 0.10, 0.95)
        density = realistic_bin_density(rng, avg=avg_density)

        steps_data.append({
            "step": i,
            "observation": {
                "base_cpu_demand":    round(cpu, 2),
                "base_mem_demand":    round(mem, 2),
                "base_latency_ms":    round(p99, 2),
                "base_error_rate":    round(clamp(base_error, 0.0, 1.0), 4),
                "base_steal_pct":     round(steal, 4),
                "active_replicas":    active_replicas,
                "buffer_depth":       buf,
                "node_size_class":    node_size,
                "current_hourly_cost": cost,
                "node_bin_density":   density,
            },
            "dynamics": {"reason": "cold_start_resource_buildout"},
        })

    return steps_data


# ── Efficient-squeeze traces ───────────────────────────────────────────────────
#
# Physics: fully provisioned cluster (5+ replicas) on a 24-hour demand cycle.
# Agent must rightsize to avoid over-provisioning cost while keeping SLA.
# - CPU and memory are deliberately decoupled (different periods + offsets)
# - Steal follows daily pattern (business hours = more noisy neighbors)
# - Error correlates with steal pressure
# - Cost responds to replica count (agent can reduce it by scaling down)
#
# Variant differences:
#   baseline    — moderate steal swing, 5 replicas on S nodes
#   steady      — very consistent, low steal; easy to rightsize
#   oscillating — aggressive steal swings; hard to keep SLA
#   gradual     — over-provisioned (7 replicas, M node); demand declining
#   optimized   — already lean (3 replicas, S node); low steal

_SQUEEZE_VARIANTS = {
    "baseline":   dict(replicas=5, steal_base=0.12, steal_amp=0.08, cpu_amp=25, mem_amp=10, node="S", cost_per_rep=1.0),
    "steady":     dict(replicas=5, steal_base=0.09, steal_amp=0.03, cpu_amp=12, mem_amp=6,  node="S", cost_per_rep=1.0),
    "oscillating":dict(replicas=5, steal_base=0.14, steal_amp=0.16, cpu_amp=30, mem_amp=15, node="S", cost_per_rep=1.0),
    "gradual":    dict(replicas=7, steal_base=0.11, steal_amp=0.06, cpu_amp=20, mem_amp=12, node="M", cost_per_rep=2.0),
    "optimized":  dict(replicas=3, steal_base=0.07, steal_amp=0.04, cpu_amp=15, mem_amp=5,  node="S", cost_per_rep=1.0),
}


def generate_squeeze(variant: str, steps: int, rng: random.Random) -> list[dict]:
    p = _SQUEEZE_VARIANTS[variant]
    active_replicas = p["replicas"]
    steps_data = []

    for i in range(steps):
        t = i / max(steps - 1, 1)

        # CPU: 24h sinusoidal cycle — peak during business hours
        cpu = clamp(
            48.0 + p["cpu_amp"] * math.sin(t * 2 * math.pi + 0.5) + gn(rng, 3.5),
            0.0, 100.0,
        )

        # Memory: slower daily cycle, more stable — decoupled from CPU
        mem = clamp(
            52.0 + p["mem_amp"] * math.sin(t * 2 * math.pi + 2.1) + gn(rng, 2.0),
            0.0, 100.0,
        )

        # Steal: business-hours pattern (peak mid-day)
        steal = clamp(
            p["steal_base"] + p["steal_amp"] * math.sin(t * 2 * math.pi - 0.4)
            + gn(rng, 0.012),
            0.0, 0.45,
        )

        # Error: low baseline, proportionally higher when steal spikes
        base_error = clamp(0.008 + steal * 0.05 + gn(rng, 0.003), 0.0, 0.25)

        # Latency: steal eats CPU budget and stacks queuing delay
        p99 = clamp(
            120.0 + steal * 200.0 + (cpu / 100.0) * 50.0 + gn(rng, 10.0),
            40.0, 400.0,
        )

        # Cost: node base + per-replica cost (agent can drive this down)
        node_base = {"S": 10.0, "M": 20.0, "L": 40.0}[p["node"]]
        cost = round(
            clamp(node_base + active_replicas * p["cost_per_rep"] + gn(rng, 0.2), 0.0, 500.0),
            2,
        )

        # Buffer: proportional to steal (requests queued at stolen-CPU nodes)
        buf = clamp(int(55 + steal * 220 + rng.randint(-12, 12)), 0, 300)

        # Bin density: denser when steal is high (noisy-neighbor packs nodes harder)
        avg_density = clamp(0.50 + steal * 0.25 + gn(rng, 0.04), 0.25, 0.97)
        density = realistic_bin_density(rng, avg=avg_density)

        steps_data.append({
            "step": i,
            "observation": {
                "base_cpu_demand":     round(cpu, 2),
                "base_mem_demand":     round(mem, 2),
                "base_latency_ms":     round(p99, 2),
                "base_error_rate":     round(clamp(base_error, 0.0, 1.0), 4),
                "base_steal_pct":      round(steal, 4),
                "active_replicas":     active_replicas,
                "buffer_depth":        buf,
                "node_size_class":     p["node"],
                "current_hourly_cost": cost,
                "node_bin_density":    density,
            },
            "dynamics": {"reason": "squeeze_24h_cycle"},
        })

    return steps_data


# ── Entropy-storm traces ───────────────────────────────────────────────────────
#
# Physics: multi-tenant cluster under noisy-neighbor disruption.
# CPU cycles are stolen, causing latency spikes and cascading timeouts.
# - Steal is bursty with carry-over (a neighbor burst doesn't end instantly)
# - Error rate is directly driven by steal (stolen CPU → request timeouts)
# - Latency spikes sharply with steal
# - Memory is comparatively stable (memory is not stolen)
#
# Variant differences:
#   baseline         — moderate steal, occasional spikes
#   chaos            — high-frequency volatile steal
#   cascading        — steal grows monotonically (cascading neighbor failure)
#   reactive_failure — steal causes error bursts; partial self-recovery
#   extreme_chaos    — sustained very-high steal, latency > SLA threshold often

_ENTROPY_VARIANTS = {
    "baseline":        dict(steal_base=0.12, spike_mag=0.14, spike_prob=0.20, err_sens=0.30, carry=0.55),
    "chaos":           dict(steal_base=0.18, spike_mag=0.18, spike_prob=0.40, err_sens=0.50, carry=0.50),
    "cascading":       dict(steal_base=0.08, spike_mag=0.20, spike_prob=0.30, err_sens=0.40, carry=0.65),
    "reactive_failure":dict(steal_base=0.15, spike_mag=0.16, spike_prob=0.35, err_sens=0.65, carry=0.60),
    "extreme_chaos":   dict(steal_base=0.28, spike_mag=0.17, spike_prob=0.50, err_sens=0.75, carry=0.70),
}


def generate_entropy(variant: str, steps: int, rng: random.Random) -> list[dict]:
    p = _ENTROPY_VARIANTS[variant]
    active_replicas = 10
    node_size = "M"
    steal_carry = 0.0    # burst carry-over from previous step
    steps_data = []

    for i in range(steps):
        t = i / max(steps - 1, 1)

        # CPU: high utilisation with moderate variation
        cpu = clamp(60.0 + 18.0 * math.sin(t * 3 * math.pi) + gn(rng, 4.5), 0.0, 100.0)

        # Memory: stable — memory is not stolen by noisy neighbors
        mem = clamp(56.0 + 8.0 * math.sin(t * math.pi + 1.2) + gn(rng, 2.0), 0.0, 100.0)

        # Steal: bursty Poisson-like, with decaying carry-over
        steal_base = p["steal_base"]
        if variant == "cascading":
            steal_base += t * 0.22   # growing neighbor failure over time

        if rng.random() < p["spike_prob"]:
            spike = p["spike_mag"] * (0.8 + rng.random() * 0.7)
            steal_carry = spike
        else:
            steal_carry *= p["carry"]   # exponential decay between bursts

        steal = clamp(steal_base + steal_carry + gn(rng, 0.015), 0.0, 0.50)

        # Error: timeout-driven; rises sharply when steal is high
        base_error = clamp(
            0.008 + steal * p["err_sens"] + gn(rng, 0.010),
            0.0, 1.0,
        )

        # Latency: stolen CPU stacks queuing delay severely
        p99 = clamp(
            140.0 + steal * 380.0 + base_error * 90.0 + gn(rng, 14.0),
            40.0, 650.0,
        )

        # Cost: M node base + replicas  (agent may scale up to compensate for steal)
        cost = round(clamp(25.0 + active_replicas * 1.0 + gn(rng, 0.2), 0.0, 500.0), 2)

        # Buffer: drains slowly when CPU is stolen; spikes during bursts
        buf = clamp(int(65 + steal * 260 + rng.randint(-15, 15)), 0, 300)

        # Bin density: high-utilisation nodes get packed even harder under steal
        avg_density = clamp(0.62 + steal * 0.18 + gn(rng, 0.04), 0.35, 0.99)
        density = realistic_bin_density(rng, avg=avg_density)

        steps_data.append({
            "step": i,
            "observation": {
                "base_cpu_demand":     round(cpu, 2),
                "base_mem_demand":     round(mem, 2),
                "base_latency_ms":     round(p99, 2),
                "base_error_rate":     round(clamp(base_error, 0.0, 1.0), 4),
                "base_steal_pct":      round(steal, 4),
                "active_replicas":     active_replicas,
                "buffer_depth":        buf,
                "node_size_class":     node_size,
                "current_hourly_cost": cost,
                "node_bin_density":    density,
            },
            "dynamics": {"reason": "entropy_storm_noisy_neighbor"},
        })

    return steps_data


# ── Trace registry ─────────────────────────────────────────────────────────────
#
# Each tuple: (task_name, difficulty, filename, variant, seed)
# Seeds are distinct per file so variants are reproducible but independent.

TRACE_REGISTRY = [
    ("cold_start",        "easy",   "trace_v1_coldstart.json",              "baseline",         42),
    ("entropy_storm",     "hard",   "trace_v1_entropy.json",                "baseline",         43),
    ("efficient_squeeze", "medium", "trace_v1_squeeze.json",                "baseline",         44),
    ("cold_start",        "easy",   "trace_v2_coldstart_gradual.json",      "gradual",         101),
    ("entropy_storm",     "hard",   "trace_v2_entropy_chaos.json",          "chaos",           102),
    ("efficient_squeeze", "medium", "trace_v2_squeeze_steady.json",         "steady",          103),
    ("cold_start",        "easy",   "trace_v3_coldstart_aggressive.json",   "aggressive",      201),
    ("entropy_storm",     "hard",   "trace_v3_entropy_cascading.json",      "cascading",       202),
    ("efficient_squeeze", "medium", "trace_v3_squeeze_oscillating.json",    "oscillating",     203),
    ("cold_start",        "easy",   "trace_v4_coldstart_failed.json",       "failed",          301),
    ("entropy_storm",     "hard",   "trace_v4_entropy_reactive_failure.json","reactive_failure",302),
    ("efficient_squeeze", "medium", "trace_v4_squeeze_gradual.json",        "gradual",         303),
    ("cold_start",        "easy",   "trace_v5_coldstart_optimal.json",      "optimal",         401),
    ("entropy_storm",     "hard",   "trace_v5_entropy_extreme_chaos.json",  "extreme_chaos",   402),
    ("efficient_squeeze", "medium", "trace_v5_squeeze_optimized.json",      "optimized",       403),
]

_GENERATORS = {
    "cold_start":        generate_cold_start,
    "efficient_squeeze": generate_squeeze,
    "entropy_storm":     generate_entropy,
}


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    traces_dir = Path(os.getenv("TRACES_DIR", "traces"))
    traces_dir.mkdir(exist_ok=True)

    subset = TRACE_REGISTRY[:TRACE_MAX_COUNT]

    for task_name, difficulty, filename, variant, seed in subset:
        rng = random.Random(seed)
        print(f"Generating {filename}  (variant={variant!r}, seed={seed}) ...")

        steps_data = _GENERATORS[task_name](variant, TRACE_STEPS, rng)

        data = {
            "task_name":       task_name,
            "task_difficulty": difficulty,
            "description":     (
                f"Realistic {task_name} trace — variant: {variant} "
                f"({TRACE_STEPS} steps, seed={seed})."
            ),
            "steps": steps_data,
        }

        out_path = traces_dir / filename
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        print(f"  OK  {out_path}")

    print(
        f"\nDone. {len(subset)} traces generated "
        f"(TRACE_STEPS={TRACE_STEPS}, TRACE_MAX_COUNT={TRACE_MAX_COUNT})."
    )


if __name__ == "__main__":
    main()
