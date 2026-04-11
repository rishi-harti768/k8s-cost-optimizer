import os
import requests
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# The repo ID of your space: e.g., "username/space-name"
REPO_ID = os.environ.get("REPO_ID", "rishi-harti768/k8s-cost-optimizer")
# Your Hugging Face User Access Token (with READ permission)
# This is DIFFERENT from the sk-... token used for LLMs
HF_USER_TOKEN: str | None = os.environ.get("HF_USER_TOKEN")


def fetch_logs(repo_id: str, token: str, log_type: str = "app"):
    """
    Fetches the last few lines of logs from a HF Space.
    """
    if not token:
        print("❌ Error: HF_USER_TOKEN is not set.")
        print("Please set it in your .env file to fetch logs programmatically.")
        return

    url = f"https://huggingface.co/api/spaces/{repo_id}/logs?type={log_type}"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"🔍 Fetching {log_type} logs for {repo_id}...")

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            logs = response.text
            print("\n--- BEGIN LOGS ---")
            # If the logs are too long, just show the last 50 lines
            lines = logs.splitlines()
            if len(lines) > 50:
                print(f"... (skipping {len(lines) - 50} lines) ...")
                print("\n".join(lines[-50:]))
            else:
                print(logs)
            print("\n--- END LOGS ---")

            # Check for common error keywords
            error_keywords = [
                "Traceback",
                "ERROR",
                "RuntimeError",
                "Exception",
                "Build failed",
            ]
            found_errors = [k for k in error_keywords if k.lower() in logs.lower()]
            if found_errors:
                print(f"\n⚠️  Potential issues found: {', '.join(found_errors)}")
            else:
                print("\n✅ No obvious runtime errors found in the current logs.")

        elif response.status_code == 401:
            print(
                "❌ Unauthorized: Your HF_USER_TOKEN might be invalid or missing permissions."
            )
        elif response.status_code == 404:
            print(f"❌ Not Found: Space {repo_id} does not exist or is private.")
        else:
            print(f"❌ Failed to fetch logs: HTTP {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    fetch_logs(REPO_ID, HF_USER_TOKEN)
