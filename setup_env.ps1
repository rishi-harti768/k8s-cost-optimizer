# setup_env.ps1
# Helper script to set environment variables for KubeCost-Gym

Write-Host "KubeCost-Gym Environment Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if HF_TOKEN is already set
if ($env:HF_TOKEN) {
    Write-Host "[OK] HF_TOKEN is already set" -ForegroundColor Green
} else {
    Write-Host "[!] HF_TOKEN is not set" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please enter your API key (or 'dummy-token' for testing):" -ForegroundColor Yellow
    $token = Read-Host "HF_TOKEN"
    
    if ($token) {
        $env:HF_TOKEN = $token
        Write-Host "[OK] HF_TOKEN set for this session" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] HF_TOKEN is required" -ForegroundColor Red
        exit 1
    }
}

# Optional: Set MODEL_NAME
Write-Host ""
Write-Host "Current MODEL_NAME: $($env:MODEL_NAME)" -ForegroundColor Cyan
Write-Host "Press Enter to keep default (mistralai/Mistral-7B-Instruct-v0.2) or enter new value:"
$model = Read-Host "MODEL_NAME"
if ($model) {
    $env:MODEL_NAME = $model
    Write-Host "[OK] MODEL_NAME set to: $model" -ForegroundColor Green
}

# Optional: Set API_BASE_URL
Write-Host ""
Write-Host "Current API_BASE_URL: $($env:API_BASE_URL)" -ForegroundColor Cyan
Write-Host "Press Enter to keep default (https://api.openai.com/v1) or enter new value:"
$api_url = Read-Host "API_BASE_URL"
if ($api_url) {
    $env:API_BASE_URL = $api_url
    Write-Host "[OK] API_BASE_URL set to: $api_url" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  HF_TOKEN     : ********" -ForegroundColor White
Write-Host "  MODEL_NAME   : $($env:MODEL_NAME)" -ForegroundColor White
Write-Host "  API_BASE_URL : $($env:API_BASE_URL)" -ForegroundColor White
Write-Host ""
Write-Host "You can now run: uv run python inference.py" -ForegroundColor Cyan
Write-Host ""
