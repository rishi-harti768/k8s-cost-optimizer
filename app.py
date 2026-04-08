# app.py
# KubeCost-Gym HTTP API Server (OpenEnv Standard)
# Explicit implementation to ensure full protocol compliance and bypass library bugs.

import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Body, HTTPException
from env import KubeCostEnv
from models import Action, Observation, ActionType, EnvState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app directly
app = FastAPI(title="KubeCost-Gym OpenEnv API")

# Initialize environment state
app.state.env = None

def get_env(task_name: Optional[str] = None, trace_path: Optional[str] = None) -> KubeCostEnv:
    """Helper to get or create environment."""
    if trace_path is None:
        trace_path = "traces/trace_v1_coldstart.json"
        if task_name == "efficient_squeeze":
            trace_path = "traces/trace_v1_squeeze.json"
        elif task_name == "entropy_storm":
            trace_path = "traces/trace_v1_entropy.json"
            
    if app.state.env is None or app.state.env.trace_path != trace_path:
        logger.info(f"Creating KubeCostEnv session [task={task_name}, trace={trace_path}]")
        app.state.env = KubeCostEnv(trace_path=trace_path)
    return app.state.env

@app.post("/reset")
async def reset(payload: Dict[str, Any] = Body(default={})):
    """
    Reset environment to initial state.
    Returns: {"observation": {...}, "reward": 0.0, "done": false, "task_name": "..."}
    """
    task_name = payload.get("task_name")
    trace_path = payload.get("trace_path")
    
    env = get_env(task_name=task_name, trace_path=trace_path)
    obs = env.reset()
    
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "task_name": env.task_name
    }

@app.post("/step")
async def step(payload: Dict[str, Any] = Body(...)):
    """
    Execute action. Handles both flat and nested 'action' field.
    """
    if app.state.env is None:
        get_env().reset()
        
    # Standard OpenEnv often nests action in an "action" field
    action_data = payload.get("action", payload)
    
    try:
        # If action_data is just a string (the action type value)
        if isinstance(action_data, str):
            action = Action(action_type=ActionType(action_data))
        # If action_data is a dict containing action_type
        elif isinstance(action_data, dict) and "action_type" in action_data:
            action = Action(**action_data)
        else:
            # Fallback for unexpected formats
            raise ValueError(f"Invalid action format: {action_data}")
            
        obs, reward, done, info = app.state.env.step(action)
        
        return {
            "observation": obs.model_dump(),
            "reward": float(reward),
            "done": bool(done),
            "task_name": app.state.env.task_name,
            "info": info
        }
    except Exception as e:
        logger.error(f"Step failed: {e}")
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/state")
async def state():
    """Get current environment state."""
    if app.state.env is None:
        get_env().reset()
    
    state_obj = app.state.env.state()
    # Ensure it returns a dict as expected by FastAPI JSONResponse
    return state_obj.model_dump()

@app.get("/health")
async def health():
    return {"status": "healthy", "env": "kubecost-gym"}

@app.get("/")
async def index():
    return {
        "name": "KubeCost-Gym",
        "api": "OpenEnv REST",
        "compliance": "Deep Check v1.0"
    }

def main():
    """Uvicorn entry point."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("SERVER_NAME", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
