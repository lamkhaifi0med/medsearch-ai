# Assemble the Hugging Face Space repo in ..\medsearch-space
# Usage:  powershell -ExecutionPolicy Bypass -File deploy\space\build_space.ps1
$ErrorActionPreference = "Stop"
$root  = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent   # MEDICAL_PROJECT
$space = Join-Path (Split-Path $root -Parent) "medsearch-space"

Write-Host "Assembling Space repo at $space"
New-Item -ItemType Directory -Force -Path $space | Out-Null

# backend (code + requirements only)
robocopy "$root\backend\app" "$space\backend\app" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS | Out-Null
Copy-Item "$root\backend\requirements.txt" "$space\backend\requirements.txt" -Force

# frontend (sources only, no node_modules/dist)
robocopy "$root\frontend\src" "$space\frontend\src" /MIR /NFL /NDL /NJH /NJS | Out-Null
foreach ($f in "index.html","package.json","package-lock.json","tsconfig.json","vite.config.ts") {
    Copy-Item "$root\frontend\$f" "$space\frontend\$f" -Force
}

# data (LFS)
New-Item -ItemType Directory -Force -Path "$space\data\processed" | Out-Null
Copy-Item "$root\data\processed\cases_clean.jsonl" "$space\data\processed\cases_clean.jsonl" -Force

# space files
Copy-Item "$PSScriptRoot\Dockerfile" "$space\Dockerfile" -Force
Copy-Item "$PSScriptRoot\README.md" "$space\README.md" -Force
Set-Content "$space\.gitattributes" "*.jsonl filter=lfs diff=lfs merge=lfs -text"
Set-Content "$space\.gitignore" "node_modules/`n__pycache__/`n*.pyc"

Write-Host "Done. Contents:"
Get-ChildItem $space | Select-Object Name
