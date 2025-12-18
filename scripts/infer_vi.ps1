<#
.SYNOPSIS
    F5-TTS Vietnamese Inference Script (using config file)

.DESCRIPTION
    Run F5-TTS inference using a TOML config file to avoid UTF-8 encoding issues.

.PARAMETER Config
    Path to TOML config file. Default: configs/infer_vi.toml

.EXAMPLE
    .\infer_vi.ps1

.EXAMPLE
    .\infer_vi.ps1 -Config "configs/my_config.toml"
#>

param(
    [string]$Config = "configs\infer_vi.toml"
)

# Set encoding for UTF-8 support
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Stop"

# Set PYTHONPATH to include src directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"

# Change to project root
Set-Location $projectRoot

Write-Host "=============================================="
Write-Host "F5-TTS Vietnamese Inference"
Write-Host "=============================================="
Write-Host "Config: $Config"
Write-Host "=============================================="
Write-Host ""

# Run inference with config file
python -m f5_tts.infer.infer_cli --config $Config

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Inference failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[SUCCESS] Inference completed!" -ForegroundColor Green
