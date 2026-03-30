# Dockerfile
# KubeCost-Gym Docker Image
# Base: python:3.10-slim (spec requirement)
# Deployment: HuggingFace Spaces with cpu-basic hardware

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching optimization)
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify structure: inference.py must exist in root (spec §5)
RUN test -f inference.py || (echo "ERROR: inference.py not found in root directory" && exit 1)

# Expose port (HF Spaces standard)
EXPOSE 7860

# Default command: run inference pipeline
CMD ["python", "inference.py"]
