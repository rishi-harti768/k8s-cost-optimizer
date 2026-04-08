# server/app.py
# Re-exported for OpenEnv compliance

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to sys.path to avoid circular import with 'server/app.py' vs root 'app.py'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app

def main():
    """Server entry point for OpenEnv multi-mode."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("SERVER_NAME", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
