import json
import math
from pathlib import Path

def generate_sinusoidal_trace(task_name, difficulty, steps=50):
    steps_data = []
    for i in range(steps):
        # Programmatic sinusoidal CPU model
        # Base CPU = 40, Amplitude = 30, Period = 20 steps
        # This creates a dynamic environment that testing agents must adapt to.
        cpu_usage = 40.0 + 30.0 * math.sin(i / 20.0 * 2 * math.pi)
        
        # Memory follows CPU but with less variance
        mem_usage = 50.0 + 15.0 * math.sin(i / 20.0 * 2 * math.pi + 0.5)
        
        # Define baseline replicas for the trace.
        # These are the "default" values that the physics overlay modifies.
        if task_name == "cold_start":
            active_replicas = 0
            node_size = "S"
            # Higher initial error rate to force agent to scale up
            base_error = 0.5 if i < 5 else 0.01
        elif task_name == "efficient_squeeze":
            active_replicas = 5
            node_size = "S"
            base_error = 0.01
        else: # entropy_storm
            active_replicas = 10
            node_size = "M"
            base_error = 0.01
            
        # Periodic spikes in cpu_steal_pct for Task 3 (Entropy Storm)
        # Every 10 steps, we have a "noisy neighbor" event starting at step 5
        is_spike = (i % 10 >= 5 and i % 10 <= 7)
        steal_pct = 0.25 if (is_spike and task_name == "entropy_storm") else 0.01
            
        observation = {
            "cpu_usage_pct": round(max(0, min(100, cpu_usage)), 2),
            "mem_usage_pct": round(max(0, min(100, mem_usage)), 2),
            "p99_latency_ms": round(150.0 + i * 2.0, 2), # increasing latency floor
            "http_error_rate": base_error,
            "cpu_steal_pct": steal_pct,
            "active_replicas": active_replicas,
            "buffer_depth": 100 + i * 10,
            "node_size_class": node_size,
            "current_hourly_cost": 10.0,
            "node_bin_density": [0.5] * 10
        }
        
        steps_data.append({
            "step": i,
            "observation": observation,
            "dynamics": {"reason": "sinusoidal_load_model"}
        })
        
    return {
        "task_name": task_name,
        "task_difficulty": difficulty,
        "description": f"Programmatic {task_name} trace with {steps} steps (Sinusoidal CPU model).",
        "steps": steps_data
    }

def main():
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)

    # All 20 traces present in the directory
    trace_files = [
        ("cold_start", "easy", "trace_v1_coldstart.json"),
        ("entropy_storm", "hard", "trace_v1_entropy.json"),
        ("efficient_squeeze", "medium", "trace_v1_squeeze.json"),
        ("cold_start", "easy", "trace_v2_coldstart_gradual.json"),
        ("entropy_storm", "hard", "trace_v2_entropy_chaos.json"),
        ("efficient_squeeze", "medium", "trace_v2_squeeze_steady.json"),
        ("cold_start", "easy", "trace_v3_coldstart_aggressive.json"),
        ("entropy_storm", "hard", "trace_v3_entropy_cascading.json"),
        ("efficient_squeeze", "medium", "trace_v3_squeeze_oscillating.json"),
        ("cold_start", "easy", "trace_v4_coldstart_failed.json"),
        ("entropy_storm", "hard", "trace_v4_entropy_reactive_failure.json"),
        ("efficient_squeeze", "medium", "trace_v4_squeeze_gradual.json"),
        ("cold_start", "easy", "trace_v5_coldstart_optimal.json"),
        ("entropy_storm", "hard", "trace_v5_entropy_extreme_chaos.json"),
        ("efficient_squeeze", "medium", "trace_v5_squeeze_optimized.json"),
        ("entropy_storm", "hard", "trace_v6_entropy_node_upgrade.json"),
        ("efficient_squeeze", "medium", "trace_v6_squeeze_challenge.json"),
        ("entropy_storm", "hard", "trace_v7_entropy_degradation.json"),
        ("entropy_storm", "hard", "trace_v8_entropy_recovery.json"),
        ("entropy_storm", "hard", "trace_v9_entropy_optimal.json"),
    ]

    for task_name, diff, filename in trace_files:
        print(f"Generating {filename}...")
        data = generate_sinusoidal_trace(task_name, diff)
        with (traces_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    
    print(f"Done generating {len(trace_files)} traces.")

if __name__ == "__main__":
    main()
