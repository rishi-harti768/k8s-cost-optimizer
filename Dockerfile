# Dockerfile
# KubeCost-Gym Docker Image
# Base: python:3.10-slim (spec requirement)
# Deployment: HuggingFace Spaces with cpu-basic hardware
# Dependency Management: uv (fast, reliable Python packaging)

FROM python:3.10-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project metadata and lockfile first (layer caching optimization)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
# Use --locked to validate lockfile is up-to-date (not --frozen, which is for workspaces only)
RUN uv sync --locked

# Copy application code
COPY . .

<<<<<<< HEAD
# Verify structure: inference.py AND app.py must exist in root (spec §5)
RUN test -f inference.py || (echo "ERROR: inference.py not found in root directory" && exit 1)
RUN test -f app.py || (echo "ERROR: app.py not found in root directory" && exit 1)
=======
# Verify structure: all critical files must exist
RUN test -f inference.py || (echo "ERROR: inference.py not found in root directory" && exit 1)
RUN test -f env.py || (echo "ERROR: env.py not found in root directory" && exit 1)
RUN test -f models.py || (echo "ERROR: models.py not found in root directory" && exit 1)
RUN test -f graders.py || (echo "ERROR: graders.py not found in root directory" && exit 1)
RUN test -f server/app.py || (echo "ERROR: server/app.py not found in server directory" && exit 1)
RUN test -f openenv.yaml || (echo "ERROR: openenv.yaml not found in root directory" && exit 1)

# Verify trace files are present
RUN test -d traces && test -f traces/trace_v1_coldstart.json || (echo "ERROR: traces directory or trace_v1_coldstart.json missing" && exit 1)

# Verify openenv.yaml is valid YAML (use uv run to access virtual environment)
RUN uv run python -c "import yaml; yaml.safe_load(open('openenv.yaml'))" || (echo "ERROR: openenv.yaml is not valid YAML" && exit 1)
>>>>>>> main

# Expose port (HF Spaces standard)
EXPOSE 7860

<<<<<<< HEAD
# Default command: run FastAPI HTTP server (OpenEnv REST API)
# inference.py stays in root for static file validation; server handles live checks
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
=======
# Default command: run OpenEnv REST API server via uv
# Uses openenv-core's create_fastapi_app for automatic endpoint generation
CMD ["uv", "run", "python", "server/app.py"]
>>>>>>> main
