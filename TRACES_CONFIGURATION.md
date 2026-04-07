# Traces Configuration Guide

Traces are now fully configurable through environment variables. This allows you to:
- Change the traces directory location
- Override individual task trace paths
- Use different trace sets for different environments

## Environment Variables

### TRACES_DIR
- **Default**: `traces`
- **Description**: Root directory where trace files are stored
- **Example**: `export TRACES_DIR=/path/to/custom/traces`

### TRACE_VERSION
- **Default**: `v1`
- **Description**: Which trace version to use (v1 through v9 available)
- **Example**: `export TRACE_VERSION=v2`
- **Note**: Automatically adjusts trace filenames (e.g., `trace_v2_coldstart.json`)

### TRACE_STEPS
- **Default**: `50`
- **Description**: Number of steps per trace (used when generating new traces)
- **Example**: `export TRACE_STEPS=100`
- **Note**: Only affects `generate_traces.py` execution

### TRACE_MAX_COUNT
- **Default**: `9999` (no limit)
- **Description**: Maximum number of traces to generate (useful for testing)
- **Example**: `export TRACE_MAX_COUNT=3` (generates only first 3 traces)
- **Note**: Only affects `generate_traces.py` execution

### Task-Specific Trace Overrides
You can override the trace path for specific tasks:

- `TASK_TRACE_COLD_START`: Path to cold_start task trace
  - Default: `{TRACES_DIR}/trace_{TRACE_VERSION}_coldstart.json`
  - Example: `export TASK_TRACE_COLD_START=/custom/cold_start_v2.json`

- `TASK_TRACE_EFFICIENT_SQUEEZE`: Path to efficient_squeeze task trace
  - Default: `{TRACES_DIR}/trace_{TRACE_VERSION}_squeeze.json`
  - Example: `export TASK_TRACE_EFFICIENT_SQUEEZE=/custom/squeeze_v2.json`

- `TASK_TRACE_ENTROPY_STORM`: Path to entropy_storm task trace
  - Default: `{TRACES_DIR}/trace_{TRACE_VERSION}_entropy.json`
  - Example: `export TASK_TRACE_ENTROPY_STORM=/custom/entropy_v2.json`

## Usage Examples

### Switch Trace Versions (PowerShell)

```powershell
# Use v2 traces
$env:TRACE_VERSION = "v2"
python app.py

# Use v3 traces
$env:TRACE_VERSION = "v3"
python inference.py

# Use v5 traces with custom steps
$env:TRACE_VERSION = "v5"
$env:TRACE_STEPS = "75"
python app.py
```

### Control Trace Generation (PowerShell)

```powershell
# Generate traces with 100 steps each (instead of default 50)
$env:TRACE_STEPS = "100"
python generate_traces.py

# Generate only the first 3 traces (for testing)
$env:TRACE_MAX_COUNT = "3"
$env:TRACE_STEPS = "50"
python generate_traces.py

# Custom directory with custom step count
$env:TRACES_DIR = "C:\data\custom_traces"
$env:TRACE_STEPS = "200"
python generate_traces.py
```

### Linux/macOS Examples

```bash
# Use v2 traces
export TRACE_VERSION=v2
python app.py

# Generate 100-step traces
export TRACE_STEPS=100
python generate_traces.py

# Generate only first 5 traces with 75 steps
export TRACE_MAX_COUNT=5
export TRACE_STEPS=75
python generate_traces.py
```

### Environment File (.env)

Create a `.env` file in the root directory:

```
TRACES_DIR=custom_traces
TASK_TRACE_COLD_START=custom_traces/trace_v2_coldstart.json
TASK_TRACE_EFFICIENT_SQUEEZE=custom_traces/trace_v2_squeeze.json
TASK_TRACE_ENTROPY_STORM=custom_traces/trace_v2_entropy.json
```

The variables will be loaded automatically by the `load_env()` function in `inference.py`.

## Files Modified

- **app.py** - Added `_load_task_traces()` function to read from env vars
- **inference.py** - Added `_load_tasks()` function to read from env vars
- **generate_traces.py** - Modified to use `TRACES_DIR` env var for output location

## Behavior

1. **Default Behavior**: If no environment variables are set, traces are loaded from the `traces/` directory with v1 versions
2. **Directory Override**: Setting `TRACES_DIR` changes the base directory for all relative trace paths
3. **Individual Overrides**: You can override specific task traces while keeping others at defaults
4. **Full Path Support**: Both relative and absolute paths are supported

## Example: Using Different Trace Versions

Quickly switch between trace versions:

```powershell
# Use v1 traces (default)
python app.py

# Use v2 traces
$env:TRACE_VERSION = "v2"
python app.py

# Use v3 traces
$env:TRACE_VERSION = "v3"
python app.py

# Use v5 with custom step count
$env:TRACE_VERSION = "v5"
$env:TRACE_STEPS = "100"
python app.py
```

## Example: Generating Custom Traces

Generate new traces with different configurations:

```powershell
# Generate standard traces (50 steps each)
python generate_traces.py

# Generate traces with 100 steps each
$env:TRACE_STEPS = "100"
python generate_traces.py

# Generate only the first 3 traces for quick testing
$env:TRACE_MAX_COUNT = "3"
python generate_traces.py

# Generate 10 traces with 200 steps each
$env:TRACE_MAX_COUNT = "10"
$env:TRACE_STEPS = "200"
python generate_traces.py
```
