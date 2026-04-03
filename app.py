"""
Gradio Web Interface for KubeCost-Gym Cost Optimization Agent.

This application wraps the inference pipeline in a persistent Gradio web server
to run on Hugging Face Spaces. It provides an interactive interface to:

1. Execute inference tasks (cold_start, efficient_squeeze, entropy_storm)
2. View real-time results and scores
3. Display task descriptions and SLA constraints
4. Monitor overall performance

Entry point: gradio app.py
Runs on: port 7860 (HF Spaces default) or environment PORT variable
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import gradio as gr

from inference import (
    CostOptimizerAgent,
    EnvironmentValidationError,
    validate_env,
    _CONFIG,
)
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from env import KubeCostEnv
from models import Observation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global state
agent: CostOptimizerAgent = None
results: Dict[str, float] = {}
start_time = datetime.now()


def initialize_agent():
    """Initialize the cost optimizer agent with environment variables."""
    global agent

    try:
        validate_env()
        logger.info("✓ Environment variables validated")

        model_name = os.environ.get("MODEL_NAME")
        api_base_url = os.environ.get("API_BASE_URL")
        agent = CostOptimizerAgent(model_name=model_name, api_base_url=api_base_url)
        logger.info(f"✓ Agent initialized with model: {model_name}")
        return True, "Agent initialized successfully"
    except EnvironmentValidationError as e:
        logger.error(f"Environment validation failed: {e}")
        return False, f"❌ Environment error: {e}"
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return False, f"❌ Initialization error: {e}"


def run_single_task(task_name: str) -> Tuple[str, str, float]:
    """
    Execute a single inference task and return results.

    Args:
        task_name: Name of task (cold_start, efficient_squeeze, entropy_storm)

    Returns:
        Tuple of (status_message, detailed_results, score)
    """
    global agent, results

    if agent is None:
        return "❌ Agent not initialized", "", 0.0

    try:
        # Find task config
        task_config = next(
            (tc for tc in _CONFIG.TASK_CONFIGS if tc["name"] == task_name), None
        )
        if not task_config:
            return f"❌ Unknown task: {task_name}", "", 0.0

        logger.info(f"\nRunning task: {task_name}")

        # Initialize environment and grader
        trace_path = task_config["trace"]
        grader_cls_name = task_config["grader"]

        grader_map = {
            "ColdStartGrader": ColdStartGrader(),
            "EfficientSqueezeGrader": EfficientSqueezeGrader(),
            "EntropyStormGrader": EntropyStormGrader(),
        }

        env = KubeCostEnv(trace_path)
        grader = grader_map[grader_cls_name]

        # Run evaluation
        score = agent.evaluate_task(env, task_name, grader)
        results[task_name] = score

        # Prepare detailed results
        detailed = f"""
**Task:** {task_name.upper()}
**Description:** {task_config['description']}
**Trace:** {trace_path}
**Score:** {score:.3f}
**Status:** {'✓ PASS' if score > 0.5 else '✗ FAIL'}

Agent executed {len(env._trajectory)} steps during evaluation.
"""

        success_msg = f"✓ Task '{task_name}' completed (score: {score:.3f})"
        return success_msg, detailed.strip(), score

    except Exception as e:
        error_msg = f"Error running task '{task_name}': {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}", "", 0.0


def run_all_tasks() -> Tuple[str, str, str]:
    """
    Execute all three inference tasks sequentially.

    Returns:
        Tuple of (status, results_summary, final_verdict)
    """
    global results

    results = {}
    task_names = [tc["name"] for tc in _CONFIG.TASK_CONFIGS]

    all_messages = []
    all_results = []

    for task_name in task_names:
        status, details, score = run_single_task(task_name)
        all_messages.append(status)
        all_results.append(details)

    # Calculate overall score
    total_score = sum(results.values()) / len(results) if results else 0.0

    # Prepare summary
    results_summary = "\n\n".join(all_results)

    # Determine verdict
    passing = total_score >= _CONFIG.PASSING_SCORE_THRESHOLD
    verdict = f"""
## FINAL RESULTS

**Average Score:** {total_score:.3f}
**Threshold:** {_CONFIG.PASSING_SCORE_THRESHOLD}
**Status:** {'✓ PASSING' if passing else '✗ FAILING'}

**Task Breakdown:**
"""

    for task_name in sorted(results.keys()):
        score = results[task_name]
        status_icon = "✓" if score > 0.5 else "✗"
        verdict += f"\n- {status_icon} {task_name}: {score:.3f}"

    status_msg = " | ".join(all_messages)

    return status_msg, results_summary, verdict


def get_system_info() -> str:
    """Return system information and configuration."""
    info = f"""
## System Information

**Startup Time:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}
**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Configuration:**
- API Base URL: {os.environ.get('API_BASE_URL', '(default)')}
- Model Name: {os.environ.get('MODEL_NAME', '(default)')}
- HF Token: {'✓ Set' if os.environ.get('HF_TOKEN') else '✗ Missing'}

**Available Tasks:**
1. **cold_start** - Scale cluster from 0→5 replicas without SLA breach
2. **efficient_squeeze** - Maintain <20% steal over 24-hour load cycle
3. **entropy_storm** - Proactive REBALANCE_NODE before steal>20%

**SLA Targets:**
- p99_latency_ms: < 300ms (healthy), < 200ms (optimal)
- http_error_rate: < 0.01 (< 1%)
- cpu_steal_pct: < 0.20 (< 20%, leading indicator)

**Scripts:** [inference.py](https://huggingface.co/spaces) · [env.py](https://huggingface.co/spaces) · [models.py](https://huggingface.co/spaces)
"""
    return info.strip()


def create_interface() -> gr.Blocks:
    """Create the Gradio interface with all tabs and components."""

    with gr.Blocks(
        title="KubeCost-Gym: K8s Cost Optimization Agent",
    ) as interface:

        # Header
        gr.Markdown(
            """
# ⚙️ KubeCost-Gym: Kubernetes Cost Optimization Agent

A production-grade RL environment for learning proactive Kubernetes autoscaling strategies using LLMs.
**Target Score:** ≥0.27 Average | **Framework:** OpenEnv · Gradio · Python 3.10+
"""
        )

        with gr.Tabs():
            # ===== TAB 1: Quick Start =====
            with gr.Tab("🚀 Quick Start"):
                gr.Markdown("### One-Click Inference")
                gr.Markdown(
                    "Run all three tasks sequentially to evaluate the agent's performance."
                )

                with gr.Group():
                    run_all_btn = gr.Button(
                        "▶ Run All Tasks", scale=1, variant="primary"
                    )
                    run_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Ready",
                        lines=1,
                    )

                with gr.Group():
                    results_display = gr.Markdown(
                        "*No results yet. Click 'Run All Tasks' to start.*"
                    )
                    verdict_display = gr.Markdown()

                run_all_btn.click(
                    run_all_tasks,
                    outputs=[run_status, results_display, verdict_display],
                )

            # ===== TAB 2: Individual Tasks =====
            with gr.Tab("📊 Individual Tasks"):
                gr.Markdown("### Run Specific Tasks")
                gr.Markdown("Select and run individual tasks for detailed analysis.")

                with gr.Row():
                    with gr.Column(scale=1):
                        task_selector = gr.Radio(
                            choices=[
                                "cold_start",
                                "efficient_squeeze",
                                "entropy_storm",
                            ],
                            label="Select Task",
                            value="cold_start",
                        )
                        run_task_btn = gr.Button("▶ Run Task", variant="primary")

                    with gr.Column(scale=2):
                        task_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2,
                        )

                task_results = gr.Markdown(label="Results")
                task_score = gr.Number(
                    label="Task Score", interactive=False, value=0.0
                )

                run_task_btn.click(
                    run_single_task,
                    inputs=[task_selector],
                    outputs=[task_status, task_results, task_score],
                )

            # ===== TAB 3: System Info =====
            with gr.Tab("ℹ️ System Info"):
                gr.Markdown("### Environment Configuration")
                info_display = gr.Markdown(get_system_info())
                gr.Button("🔄 Refresh").click(
                    get_system_info, outputs=[info_display]
                )

            # ===== TAB 4: Documentation =====
            with gr.Tab("📚 Documentation"):
                gr.Markdown(
                    """
### About KubeCost-Gym

**Environment:** Simulates a production Kubernetes cluster with:
- Dynamic workloads (sinusoidal CPU/memory curves)
- Resource constraints (node capacity, cost budget)
- SLA enforcement (latency, error rate, CPU steal)

**Agent:** Uses an LLM with OpenAI Client to make cost optimization decisions.

**Tasks:**
1. **cold_start** (~easy): Scale cluster from 0 to 5+ replicas quickly
   - Priority: Reach 5+ replicas without SLA breach
   - Constraint: p99_latency_ms < 300ms

2. **efficient_squeeze** (~medium): Maintain operational cost efficiency
   - Priority: Reduce cost while keeping cpu_steal_pct < 20%
   - Constraint: Balance cost vs. reliability

3. **entropy_storm** (~hard): Proactive scaling under chaos
   - Priority: USE REBALANCE_NODE to prevent issues
   - Constraint: Be proactive, not reactive

**Scoring:** Each task is graded 0.0-1.0. Average ≥0.27 passes.

**Architecture:** 
```
User Input → Gradio Interface → Agent.decide() 
            → LLM API (OpenAI Compatible)
            → Environment.step()
            → Grader.evaluate()
            → Score + Results
```

**References:**
- [OpenEnv Spec](https://huggingface.co/docs)
- [PROJECT_SPEC.md](https://huggingface.co/spaces)
- [Kubernetes Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
"""
                )

    return interface


def main():
    """Main entry point. Initialize and launch Gradio app."""
    logger.info("=" * 70)
    logger.info("KubeCost-Gym Gradio Application Starting")
    logger.info("=" * 70)

    # Initialize agent
    success, msg = initialize_agent()
    if not success:
        logger.warning(f"Agent initialization warning: {msg}")
        # Continue anyway - agent will be initialized on first use if env vars set

    # Create interface
    interface = create_interface()

    # Get port from environment or use default
    port = int(os.environ.get("PORT", 7860))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

    logger.info(f"Launching Gradio app on {server_name}:{port}")

    # Launch with queue enabled for better handling of long-running tasks
    interface.queue(
        default_concurrency_limit=1,  # Process one task at a time
        max_size=5,  # Queue up to 5 requests
    ).launch(
        server_name=server_name,
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
        ),
    )


if __name__ == "__main__":
    main()
