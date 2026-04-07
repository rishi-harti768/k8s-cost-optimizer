# ✅ Final Configuration - Locked to v1 Traces

## Configuration Summary

### Traces (Hardcoded to v1)
```
✓ traces/trace_v1_coldstart.json  (25 steps)
✓ traces/trace_v1_squeeze.json    (25 steps)
✓ traces/trace_v1_entropy.json    (25 steps)
```

**Total: 3 traces, 75 LLM calls (25 steps × 3 tasks)**

### API Configuration (.env file)
```
HF_TOKEN     = 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
MODEL_NAME   = openai/gpt-oss-120b
API_BASE_URL = https://integrate.api.nvidia.com/v1
```

## What Changed

### Before (Configurable)
```python
TRACE_VERSION = os.getenv("TRACE_VERSION", "v1")  # Could be changed
trace = f"traces/trace_{TRACE_VERSION}_coldstart.json"
```

### After (Fixed to v1)
```python
TASKS = [
    {"trace": "traces/trace_v1_coldstart.json", ...},
    {"trace": "traces/trace_v1_squeeze.json", ...},
    {"trace": "traces/trace_v1_entropy.json", ...},
]
```

## Why v1 Only?

1. **Official Grading** - v1 traces are used for final scoring
2. **Consistency** - Everyone tests on the same traces
3. **Simplicity** - No confusion about which version to use
4. **Deployment Ready** - HuggingFace Spaces will use v1

## Running Inference

### Simple Command
```bash
uv run python inference.py
```

### What Happens
```
1. Loads .env file (NVIDIA API credentials)
2. Runs cold_start task (25 steps)
3. Runs efficient_squeeze task (25 steps)
4. Runs entropy_storm task (25 steps)
5. Shows final scores
```

### Expected Runtime
```
Total LLM calls: 75
Estimated time:  ~2.5 minutes
API provider:    NVIDIA (gpt-oss-120b)
```

## Testing Other Versions (Optional)

If you want to test v2-v5 traces for development:

### Option 1: Temporarily Edit inference.py
```python
# Change the trace paths manually
TASKS = [
    {"trace": "traces/trace_v2_coldstart.json", ...},  # v2 instead of v1
    ...
]
```

### Option 2: Create a Test Script
```python
# test_all_versions.py
from inference import CostOptimizerAgent
from env import KubeCostEnv
from graders import ColdStartGrader

agent = CostOptimizerAgent()

for version in ['v1', 'v2', 'v3', 'v4', 'v5']:
    trace = f"traces/trace_{version}_coldstart.json"
    env = KubeCostEnv(trace)
    # Run test...
    print(f"{version}: score = ...")
```

## Validation

```bash
$ python validate_local.py
✓ All 6/6 checks passed
```

## Files Status

### Production Files (v1 only)
- ✅ `inference.py` - Hardcoded to v1 traces
- ✅ `.env` - NVIDIA API configuration
- ✅ `traces/trace_v1_*.json` - Official grading traces (3 files)

### Development Files (v2-v5 available)
- ✅ `traces/trace_v2_*.json` - Easier variants (3 files)
- ✅ `traces/trace_v3_*.json` - Harder variants (3 files)
- ✅ `traces/trace_v4_*.json` - Failure scenarios (3 files)
- ✅ `traces/trace_v5_*.json` - Optimal scenarios (3 files)

**Total: 15 trace files (3 for production, 12 for development)**

## Deployment Checklist

When deploying to HuggingFace Spaces:

- [x] inference.py uses v1 traces (hardcoded)
- [x] .env file configured with NVIDIA API
- [x] All 3 v1 trace files present
- [x] Each trace has 25 steps
- [x] Validation passes (6/6 checks)

## Quick Reference

| Item | Value |
|------|-------|
| Traces Used | v1 only (official) |
| Total Traces | 3 files |
| Steps per Trace | 25 |
| Total LLM Calls | 75 |
| Runtime | ~2.5 minutes |
| API Provider | NVIDIA |
| Model | openai/gpt-oss-120b |

---

**Status**: ✅ Locked to v1 traces
**Ready**: ✅ Yes
**Command**: `uv run python inference.py`
