# server/app.py
# Re-exported for OpenEnv compliance

import uvicorn
import os
from app import app

def main():
    """Server entry point for OpenEnv multi-mode."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("SERVER_NAME", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
