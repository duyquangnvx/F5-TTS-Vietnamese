<#
.SYNOPSIS
    F5-TTS Inference Script for Windows (PowerShell)

.DESCRIPTION
    Run F5-TTS inference with customizable parameters.

.PARAMETER RefAudio
    Path to reference audio file

.PARAMETER RefText
    Reference text (transcript of reference audio)

.PARAMETER GenText
    Text to generate speech for

.PARAMETER Model
    Model name (F5TTS_Base, F5TTS_v1_Base, E2TTS_Base)

.PARAMETER Speed
    Speech speed (0.5-2.0). Default: 1.0

.EXAMPLE
    .\infer.ps1 -RefAudio "ref.wav" -RefText "Hello world" -GenText "This is a test"

.EXAMPLE
    .\infer.ps1 -GenText "Xin chào các bạn" -Model "F5TTS_v1_Base" -Speed 0.9
#>

param(
    [string]$RefAudio = "ref.wav",
    [string]$RefText = "cả hai bên hãy cố gắng hiểu cho nhau",
    [Parameter(Mandatory=$false)]
    [string]$GenText = "mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn, sau đó mang những gì học được về việt nam giúp xây dựng các công trình tốt hơn",
    [string]$Model = "F5TTS_Base",
    [string]$VocabFile = "data\your_training_dataset\vocab.txt",
    [string]$CkptFile = "ckpts\your_training_dataset\model_last.pt",
    [string]$Vocoder = "vocos",
    [float]$Speed = 1.0,
    [int]$NfeStep = 32,
    [string]$OutputDir = "output\inference"
)

# Set encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

# Set PYTHONPATH to include src directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$env:PYTHONPATH = "$projectRoot\src;$env:PYTHONPATH"

# Change to project root
Set-Location $projectRoot

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# Generate timestamp for output file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = Join-Path $OutputDir "output_$timestamp.wav"

Write-Host "=============================================="
Write-Host "F5-TTS Inference"
Write-Host "=============================================="
Write-Host "Model:      $Model"
Write-Host "Checkpoint: $CkptFile"
Write-Host "Vocab:      $VocabFile"
Write-Host "Ref Audio:  $RefAudio"
Write-Host "Speed:      $Speed"
Write-Host "NFE Steps:  $NfeStep"
Write-Host "Output Dir: $OutputDir"
Write-Host "=============================================="
Write-Host ""

# Run inference
python -m f5_tts.infer.infer_cli `
    --model $Model `
    --ref_audio $RefAudio `
    --ref_text $RefText `
    --gen_text $GenText `
    --speed $Speed `
    --nfe_step $NfeStep `
    --vocoder_name $Vocoder `
    --vocab_file $VocabFile `
    --ckpt_file $CkptFile `
    --output_dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Inference failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[SUCCESS] Inference completed!" -ForegroundColor Green
Write-Host "Output saved to: $OutputDir"
