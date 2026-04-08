import os
import requests
import json
import time
from typing import Dict, Any

# --- CONFIGURATION ---
# Replace with your Space URL or set the SPACE_URL environment variable
# Example: https://rishi-harti768-kubecost-gym.hf.space
SPACE_URL = os.environ.get("SPACE_URL", "https://rishi-harti768-k8s-cost-optimizer.hf.space").rstrip("/")

def log_test_step(name: str, status: bool, info: str = ""):
    icon = "✅" if status else "❌"
    print(f"{icon} {name:20} | {info}")

def test_remote_server():
    print(f"\n🚀 Starting Remote Server Verification for: {SPACE_URL}\n")
    
    # 1. Health Check
    try:
        resp = requests.get(f"{SPACE_URL}/health", timeout=10)
        if resp.status_code == 200:
            log_test_step("Health Check", True, f"HTTP {resp.status_code} - OK")
        else:
            log_test_step("Health Check", False, f"HTTP {resp.status_code} - {resp.text}")
            return
    except Exception as e:
        log_test_step("Health Check", False, f"Error: {str(e)}")
        return

    # 2. Reset Environment
    try:
        resp = requests.post(f"{SPACE_URL}/reset", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            obs = data.get("observation", {})
            log_test_step("Reset Env", True, f"Episode ID: {data.get('episode_id', 'N/A')}")
            # print(f"   Initial Observation: {json.dumps(obs, indent=2)}")
        else:
            log_test_step("Reset Env", False, f"HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        log_test_step("Reset Env", False, f"Error: {str(e)}")

    # 3. Get Current State
    try:
        resp = requests.get(f"{SPACE_URL}/state", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            log_test_step("Get State", True, f"Step: {data.get('step_count', 0)}, Replicas: {data.get('replicas', 'N/A')}")
        else:
            log_test_step("Get State", False, f"HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        log_test_step("Get State", False, f"Error: {str(e)}")

    # 4. Perform a Step
    # Scaling replicas up by 5
    action_payload = {
        "action": {
            "action_type": "SCALE_REPLICAS",
            "value": 5
        }
    }
    try:
        resp = requests.post(
            f"{SPACE_URL}/step", 
            json=action_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if resp.status_code == 200:
            result = resp.json()
            log_test_step("Step Execution", True, f"Reward: {result.get('reward', 0.0):.2f}")
        else:
            log_test_step("Step Execution", False, f"HTTP {resp.status_code} - {resp.text}")
            # Try to see if there's a traceback in the response
            if "traceback" in resp.text.lower() or "error" in resp.text.lower():
                print("\n⚠️ Potential Runtime Error Detected in Server Response:")
                print(resp.text)
    except Exception as e:
        log_test_step("Step Execution", False, f"Error: {str(e)}")

    # 5. Verify API docs
    try:
        resp = requests.get(f"{SPACE_URL}/docs", timeout=5)
        log_test_step("API Docs", resp.status_code == 200, f"HTTP {resp.status_code}")
    except:
        log_test_step("API Docs", False, "Timeout or Connection Error")

    print("\n--- Verification Complete ---\n")

if __name__ == "__main__":
    test_remote_server()
