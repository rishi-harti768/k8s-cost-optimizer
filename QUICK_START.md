# Quick Start Guide

## Setup

### Step 1: Configure Environment Variables

**Option A: Use .env file (Recommended)**

1. Edit the `.env` file in the project root:
   ```bash
   # Open .env and replace 'your-api-key-here' with your actual key
   HF_TOKEN=your-actual-api-key
   MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
   API_BASE_URL=https://api.openai.com/v1
   ```

2. See `.env.example` for more provider examples

**Option B: Set Environment Variables Manually**

**Option B: Set Environment Variables Manually**

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "your-api-key-here"
uv run python inference.py
```

**Windows (CMD):**
```cmd
set HF_TOKEN=your-api-key-here
uv run python inference.py
```

**Linux/Mac:**
```bash
export HF_TOKEN="your-api-key-here"
uv run python inference.py
```

### Option 2: Use Setup Script (Windows PowerShell)

```powershell
# Run the setup script
. .\setup_env.ps1

# Then run inference
uv run python inference.py
```

### Option 3: For Testing (No Real API Key)

```powershell
# Use dummy token for testing (will fail at LLM calls but validates setup)
$env:HF_TOKEN = "dummy-token"
uv run python inference.py
```

### Step 2: Run Inference

Once your environment is configured:

```bash
# Using uv (recommended)
uv run python inference.py

# Or using regular python
python inference.py
```

## Configuration Options

### Required
- **HF_TOKEN**: Your API key (OpenAI, HuggingFace, etc.)

### Optional (have defaults)
- **MODEL_NAME**: Default is `mistralai/Mistral-7B-Instruct-v0.2`
- **API_BASE_URL**: Default is `https://api.openai.com/v1`

## Examples

### Using OpenAI GPT-4
```powershell
$env:HF_TOKEN = "sk-your-openai-key"
$env:MODEL_NAME = "gpt-4"
$env:API_BASE_URL = "https://api.openai.com/v1"
uv run python inference.py
```

### Using HuggingFace Models
```powershell
$env:HF_TOKEN = "hf_your-huggingface-token"
$env:MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
$env:API_BASE_URL = "https://api-inference.huggingface.co/models"
uv run python inference.py
```

### Using NVIDIA API
```powershell
$env:HF_TOKEN = "nvapi-your-nvidia-key"
$env:MODEL_NAME = "openai/gpt-oss-120b"
$env:API_BASE_URL = "https://integrate.api.nvidia.com/v1"
uv run python inference.py
```

## Troubleshooting

### Error: "HF_TOKEN environment variable is required"
**Solution:** Set the HF_TOKEN environment variable before running:
```powershell
$env:HF_TOKEN = "your-api-key"
```

### Error: "VIRTUAL_ENV=venv does not match"
**Solution:** This is just a warning. You can:
1. Ignore it (it won't affect execution)
2. Or deactivate venv and use uv's managed environment:
   ```powershell
   deactivate
   uv run python inference.py
   ```

### Error: "Empty text from API"
**Solution:** Your model might not support JSON mode. Try:
1. Use a different model (e.g., gpt-4)
2. Or check your API endpoint is correct

## Expected Output

```
[INFO] API_BASE_URL : https://api.openai.com/v1
[INFO] MODEL_NAME   : mistralai/Mistral-7B-Instruct-v0.2
[INFO] HF_TOKEN     : ******** (hidden)
[START] {"task": "cold_start", "model": "...", "max_steps": 200}
[STEP]  {"task": "cold_start", "step": 1, "action": "...", ...}
...
[END]   {"task": "cold_start", "score": 0.85, "total_steps": 25, "status": "success"}
...
============================================================
INFERENCE RESULTS SUMMARY
============================================================
  [PASS] cold_start: 0.8500
  [PASS] efficient_squeeze: 0.7200
  [PASS] entropy_storm: 0.6100

  Average score : 0.7267
============================================================
```

## Next Steps

1. **Test locally** with your API key
2. **Review scores** in the summary
3. **Deploy to HuggingFace Spaces** when ready
4. **Set secrets** in Space settings (HF_TOKEN, MODEL_NAME, API_BASE_URL)

---

**Need help?** Check `FINAL_CONFIGURATION.md` for detailed setup information.
