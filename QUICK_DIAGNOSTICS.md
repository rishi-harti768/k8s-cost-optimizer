# Quick Diagnostics

## Test That Everything Works Locally

### 1. Verify openai is installed and working
```bash
python -c "from openai import OpenAI; print('openai: OK')"
```

### 2. Test inference.py imports
```bash
python -c "
import os
os.environ['HF_TOKEN'] = 'test'
os.environ['API_BASE_URL'] = 'https://api.openai.com/v1'
os.environ['MODEL_NAME'] = 'test'
from inference import CostOptimizerAgent
print('inference.py: OK')
"
```

### 3. Verify uv.lock is valid
```bash
cat uv.lock | head -20
grep "name = \"openai\"" uv.lock
```

### 4. Check pyproject.toml dependencies
```bash
grep "dependencies" pyproject.toml -A 10
```

### 5. Verify git changes
```bash
git log -1 --oneline
git diff HEAD~1 uv.lock | head -50
```

## If Phase 2 Still Fails

### Debug the Docker build
```bash
# Build locally
docker build -t test-build .

# If fails, check:
# 1. Dockerfile is correct
# 2. All source files exist (inference.py, app.py, env.py, etc.)
# 3. Traces directory has all JSON files
# 4. openenv.yaml exists and is valid YAML
```

### Check what's in the lock file
```bash
# Count packages
wc -l uv.lock

# Find specific package
grep "name = \"fastapi\"" uv.lock

# List all top-level dependencies
grep "^name = " uv.lock | head -20
```

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'openai'"
**Fix**: You already fixed this by regenerating uv.lock ✓

### Issue: Docker build fails during `uv sync`
**Solution**: 
1. Make sure uv.lock is valid: `uv lock --check`
2. Regenerate it: `uv lock --upgrade`
3. Commit and push

### Issue: Import works locally but fails in Docker
**Solution**:
1. Docker uses uv.lock, not your local environment
2. Always regenerate lock after changing pyproject.toml
3. Test the Dockerfile locally if possible

---

All checks have been completed and committed! ✓
