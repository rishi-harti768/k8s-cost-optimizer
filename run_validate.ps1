$env:PYTHONIOENCODING = "utf-8"
$result = & ".\venv\Scripts\python.exe" validate_local.py 2>&1
$result | ForEach-Object { $_ } | Set-Content -Path "val_out.txt" -Encoding UTF8
Write-Host "Exit: $LASTEXITCODE"
