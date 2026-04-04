<<<<<<< HEAD
# app.py
"""
OpenEnv HTTP API Server for KubeCost-Gym.

Exposes KubeCostEnv methods as REST endpoints so the OpenEnv automated
checker can interact with the environment over HTTP.

Endpoints:
    POST /reset        → Reset environment, returns initial Observation JSON
    POST /step         → Execute action, returns (obs, reward, done, info)
    GET  /state        → Current EnvState JSON
    GET  /openenv      → openenv.yaml parsed as JSON
    GET  /health       → Liveness probe

Port: 7860 (HuggingFace Spaces standard)

Spec Reference: PROJECT_SPEC.md §2 OpenEnv Interface, §5 Infra Spec
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
=======
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
>>>>>>> main
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

<<<<<<< HEAD
from env import KubeCostEnv
from models import Action, ActionType, Observation, EnvState

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KubeCost-Gym",
    description="Kubernetes cost optimization RL environment (OpenEnv standard).",
    version="3.0",
)

# ---------------------------------------------------------------------------
# Task → trace file mapping
# ---------------------------------------------------------------------------

TASK_TRACES: Dict[str, str] = {
    "cold_start":        "traces/trace_v1_coldstart.json",
    "efficient_squeeze": "traces/trace_v1_squeeze.json",
    "entropy_storm":     "traces/trace_v1_entropy.json",
}

DEFAULT_TASK = "cold_start"

# ---------------------------------------------------------------------------
# Singleton environment state (one per running container)
# ---------------------------------------------------------------------------

_env: Optional[KubeCostEnv] = None
_current_task: str = DEFAULT_TASK


def _get_env(task_name: str = DEFAULT_TASK) -> KubeCostEnv:
    """Return (or initialise) the environment for the requested task."""
    global _env, _current_task

    # Re-create if task changed or env not yet created
    if _env is None or _current_task != task_name:
        trace_path = TASK_TRACES.get(task_name, TASK_TRACES[DEFAULT_TASK])
        _env = KubeCostEnv(trace_path)
        _current_task = task_name

    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = DEFAULT_TASK


class StepRequest(BaseModel):
    action_type: str   # Must be a valid ActionType value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: Observation) -> Dict[str, Any]:
    """Serialise Pydantic Observation to a plain dict (JSON-safe)."""
    d = obs.model_dump()
    # node_size_class is an Enum — serialise to its string value
    if hasattr(d.get("node_size_class"), "value"):
        d["node_size_class"] = d["node_size_class"].value
    else:
        d["node_size_class"] = str(d["node_size_class"])
    return d


def _state_to_dict(state: EnvState) -> Dict[str, Any]:
    """Serialise Pydantic EnvState to a plain dict (JSON-safe)."""
    d = state.model_dump()
    if hasattr(d.get("node_size"), "value"):
        d["node_size"] = d["node_size"].value
    else:
        d["node_size"] = str(d["node_size"])
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe — always returns 200 OK."""
    return JSONResponse({"status": "ok", "env": "kubecost-gym", "version": "3.0"})


@app.get("/openenv")
async def get_openenv() -> JSONResponse:
    """
    Return parsed openenv.yaml as JSON.

    The OpenEnv checker calls this to verify task definitions.
    """
    yaml_path = Path("openenv.yaml")
    if not yaml_path.exists():
        raise HTTPException(status_code=500, detail="openenv.yaml not found")

    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return JSONResponse(data)


@app.post("/reset")
async def reset(request: ResetRequest = None) -> JSONResponse:
    """
    Reset the environment to its initial state.

    Body (optional JSON):
        { "task_name": "cold_start" | "efficient_squeeze" | "entropy_storm" }

    Returns:
        Initial Observation as JSON.
    """
    task_name = DEFAULT_TASK
    if request is not None and request.task_name:
        task_name = request.task_name

    if task_name not in TASK_TRACES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid tasks: {list(TASK_TRACES.keys())}"
        )

    try:
        env = _get_env(task_name)
        obs: Observation = env.reset()
        return JSONResponse({
            "observation": _obs_to_dict(obs),
            "task_name": task_name,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
async def step(request: StepRequest) -> JSONResponse:
    """
    Execute one environment step.

    Body (JSON):
        { "action_type": "<ActionType value>" }

    Returns:
        {
            "observation": {...},
            "reward": float,
            "done": bool,
            "info": {...}
        }
    """
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first."
        )

    # Validate action_type
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        valid = [a.value for a in ActionType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{request.action_type}'. Valid values: {valid}"
        ) from None

    try:
        action = Action(action_type=action_type)
        obs, reward, done, info = _env.step(action)

        # Ensure info is serialisable
        serialisable_info = {}
        for k, v in info.items():
            try:
                json.dumps(v)
                serialisable_info[k] = v
            except (TypeError, ValueError):
                serialisable_info[k] = str(v)

        return JSONResponse({
            "observation": _obs_to_dict(obs),
            "reward": float(reward),
            "done": bool(done),
            "info": serialisable_info,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
async def get_state() -> JSONResponse:
    """
    Return the current environment state snapshot.

    Returns:
        EnvState as JSON.
    """
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first."
        )

    try:
        state: EnvState = _env.state()
        return JSONResponse(_state_to_dict(state))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Root redirect → health
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "name": "kubecost-gym",
        "version": "3.0",
        "endpoints": ["/reset", "/step", "/state", "/openenv", "/health"],
    })


# ---------------------------------------------------------------------------
# Entry point (for local testing without uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
=======
from inference import (
    CostOptimizerAgent,
    EnvironmentValidationError,
    validate_env,
    _CONFIG,
)
from graders import ColdStartGrader, EfficientSqueezeGrader, EntropyStormGrader
from env import KubeCostEnv
from models import Observation, Action, ActionType

# REST API request models
class StepRequest(BaseModel):
    action: str  # e.g., "SCALE_REPLICAS(+5)"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Global state
agent: CostOptimizerAgent = None
results: Dict[str, float] = {}
start_time = datetime.now()

# REST API state - separate from Gradio agent
rest_env: KubeCostEnv = None
rest_current_trace: str = None


# ===== FastAPI REST API Setup =====

def create_fastapi_app() -> FastAPI:
    """Create FastAPI app with OpenEnv REST endpoints."""
    app = FastAPI(
        title="KubeCost-Gym OpenEnv API",
        description="REST API for KubeCost-Gym environment validation",
        version="3.0",
    )

    @app.post("/reset")
    @app.get("/reset")
    async def reset_env(trace_path: str = "traces/trace_v1_coldstart.json"):
        """
        Reset environment to initial state.
        
        Accepts:
            - POST with optional JSON body: {"trace_path": "traces/..."}
            - GET with optional query param: ?trace_path=traces/...
            - Empty request (uses default trace)
        
        Returns:
            Initial Observation as JSON
        """
        global rest_env, rest_current_trace
        
        try:
            # Create new environment
            rest_env = KubeCostEnv(trace_path)
            rest_current_trace = trace_path
            
            # Get initial observation
            obs = rest_env.reset()
            
            logger.info(f"✓ Environment reset with trace: {trace_path}")
            return obs.model_dump()
        
        except FileNotFoundError as e:
            logger.error(f"Trace file not found: {trace_path}")
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_path}")
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

    @app.post("/step")
    async def step_env(request: StepRequest):
        """
        Execute one environment step with given action.
        
        Request body:
            {"action": "SCALE_REPLICAS(+5)"}
        
        Returns:
            JSON with keys: observation, reward, done, info
        """
        global rest_env
        
        if rest_env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        try:
            # Convert action string to ActionType enum
            action_type = ActionType(request.action)
            action_obj = Action(action_type=action_type)
            
            # Execute step
            obs, reward, done, info = rest_env.step(action_obj)
            
            logger.info(f"Step executed: action={request.action}, reward={reward:.2f}, done={done}")
            
            return {
                "observation": obs.model_dump(),
                "reward": float(reward),
                "done": bool(done),
                "info": info,
            }
        
        except ValueError as e:
            logger.error(f"Invalid action: {request.action}")
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        except Exception as e:
            logger.error(f"Step failed: {e}")
            raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")

    @app.get("/state")
    async def get_state():
        """
        Get current environment state.
        
        Returns:
            Current state as JSON
        """
        global rest_env
        
        if rest_env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        try:
            state = rest_env.state()
            logger.info("State retrieved")
            return state.model_dump() if hasattr(state, 'model_dump') else state
        
        except Exception as e:
            logger.error(f"State retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"State error: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "env_initialized": rest_env is not None,
            "trace": rest_current_trace,
        }

    @app.get("/")
    async def root():
        """Root endpoint with API documentation."""
        return {
            "name": "KubeCost-Gym OpenEnv API",
            "version": "3.0",
            "description": "REST API for Kubernetes cost optimization RL environment",
            "endpoints": {
                "POST /reset": "Reset environment to initial state",
                "POST /step": "Execute action and get next step",
                "GET /state": "Get current environment state",
                "GET /health": "Health check",
                "GET /docs": "Interactive API documentation (Swagger UI)",
            },
            "docs": "/docs",
        }

    return app


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
    """Main entry point. Launch FastAPI with REST endpoints."""
    logger.info("=" * 70)
    logger.info("KubeCost-Gym Application Starting")
    logger.info("=" * 70)

    # Initialize agent
    success, msg = initialize_agent()
    if not success:
        logger.warning(f"Agent initialization warning: {msg}")
        # Continue anyway - agent will be initialized on first use if env vars set

    # Create FastAPI app with REST API endpoints
    # This is required for OpenEnv submission spec: POST /reset must be available
    logger.info("Creating FastAPI REST API endpoints...")
    app = create_fastapi_app()

    # Get port and host from environment
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("SERVER_NAME", "0.0.0.0")

    logger.info("=" * 70)
    logger.info(f"Launching FastAPI on {host}:{port}")
    logger.info("=" * 70)
    logger.info("✓ OpenEnv REST Endpoints (required by submission):")
    logger.info("  - POST /reset   - Reset environment to initial state")
    logger.info("  - POST /step    - Execute action")
    logger.info("  - GET  /state   - Get environment state")
    logger.info("  - GET  /health  - Health check")
    logger.info("  - GET  /docs    - Interactive API docs (Swagger UI)")
    logger.info("  - GET  /        - API information")

    # Launch using uvicorn
    logger.info(f"Starting server...")
    import uvicorn
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
>>>>>>> main
