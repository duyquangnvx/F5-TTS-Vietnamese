<#
.SYNOPSIS
    F5-TTS Fine-tuning Pipeline for Windows (PowerShell)

.DESCRIPTION
    Complete pipeline for fine-tuning F5-TTS models including:
    - Sample rate conversion
    - Metadata preparation
    - Vocabulary checking and extension
    - Feature extraction
    - Model fine-tuning

.PARAMETER Stage
    Starting stage (0-5). Default: 5

.PARAMETER StopStage
    Ending stage (0-5). Default: 5

.PARAMETER DatasetName
    Name of the training dataset. Default: your_training_dataset

.PARAMETER ExpName
    Experiment/model name. Default: F5TTS_Base

.EXAMPLE
    .\fine_tuning.ps1 -Stage 0 -StopStage 5

.EXAMPLE
    .\fine_tuning.ps1 -Stage 5 -DatasetName "my_dataset" -ExpName "F5TTS_v1_Base"
#>

param(
    [int]$Stage = 5,
    [int]$StopStage = 5,
    [string]$DatasetName = "your_training_dataset",
    [string]$ExpName = "F5TTS_Base",
    [int]$BatchSize = 7000,
    [int]$NumWorkers = 16,
    [int]$WarmupUpdates = 20000,
    [int]$SaveUpdates = 10000,
    [int]$LastUpdates = 10000,
    [string]$PretrainCkpt = ""
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

# Configuration
$env:CUDA_VISIBLE_DEVICES = "0"

$DatasetDir = "data\$DatasetName"
$RawDatasetDir = "data\your_dataset"
$OutputDir = "output"
$CkptsDir = "ckpts\$DatasetName"

if ([string]::IsNullOrEmpty($PretrainCkpt)) {
    $PretrainCkpt = "$CkptsDir\pretrained_model_1200000.pt"
}

# Helper function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message"
}

# Create directories
Write-Log "Creating directories..."
$dirs = @($DatasetDir, $RawDatasetDir, $OutputDir, "$OutputDir\logs", $CkptsDir)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# =============================================================================
# Stage 0: Convert sample rate to 24kHz
# =============================================================================
if ($Stage -le 0 -and $StopStage -ge 0) {
    Write-Log "Stage 0: Converting sample rate to 24kHz..."

    python -m f5_tts.tools.convert_sr `
        --input-dir $RawDatasetDir `
        --pattern "*.wav" `
        --sample-rate 24000 `
        --workers $NumWorkers `
        --keep-original

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Sample rate conversion failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 0 completed."
}

# =============================================================================
# Stage 1: Prepare metadata
# =============================================================================
if ($Stage -le 1 -and $StopStage -ge 1) {
    Write-Log "Stage 1: Preparing metadata..."

    python -m f5_tts.tools.prepare_metadata `
        --dataset-dir $RawDatasetDir `
        --training-dir $DatasetDir

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Metadata preparation failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 1 completed."
}

# =============================================================================
# Stage 2: Check vocabulary
# =============================================================================
if ($Stage -le 2 -and $StopStage -ge 2) {
    Write-Log "Stage 2: Checking vocabulary..."

    python -m f5_tts.tools.check_vocab `
        --pretrained-vocab "data\Emilia_ZH_EN_pinyin\vocab.txt" `
        --dataset-vocab "$DatasetDir\vocab.txt" `
        --output "$DatasetDir\vocab_extended.txt"

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Vocabulary check failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 2 completed."
}

# =============================================================================
# Stage 3: Extend model embeddings
# =============================================================================
if ($Stage -le 3 -and $StopStage -ge 3) {
    Write-Log "Stage 3: Extending model embeddings..."

    python -m f5_tts.tools.extend_embeddings `
        --model F5TTS_Base `
        --pretrained-vocab "data\Emilia_ZH_EN_pinyin\vocab.txt" `
        --new-vocab "$DatasetDir\vocab_extended.txt" `
        --output $PretrainCkpt

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Embedding extension failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 3 completed."
}

# =============================================================================
# Stage 4: Feature extraction
# =============================================================================
if ($Stage -le 4 -and $StopStage -ge 4) {
    Write-Log "Stage 4: Extracting features..."

    python src\f5_tts\train\datasets\prepare_csv_wavs.py `
        $DatasetDir $DatasetDir `
        --workers $NumWorkers

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Feature extraction failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 4 completed."
}

# =============================================================================
# Stage 5: Fine-tuning
# =============================================================================
if ($Stage -le 5 -and $StopStage -ge 5) {
    Write-Log "Stage 5: Starting fine-tuning..."
    Write-Log "  Experiment: $ExpName"
    Write-Log "  Dataset: $DatasetName"
    Write-Log "  Batch Size: $BatchSize"
    Write-Log "  Pretrained: $PretrainCkpt"

    # Single GPU training
    python src\f5_tts\train\finetune_cli.py `
        --exp_name $ExpName `
        --dataset_name $DatasetName `
        --batch_size_per_gpu $BatchSize `
        --num_warmup_updates $WarmupUpdates `
        --save_per_updates $SaveUpdates `
        --last_per_updates $LastUpdates `
        --finetune `
        --log_samples `
        --pretrain $PretrainCkpt

    # Multi-GPU training (uncomment to use)
    # accelerate launch src\f5_tts\train\finetune_cli.py `
    #     --exp_name $ExpName `
    #     --dataset_name $DatasetName `
    #     --batch_size_per_gpu $BatchSize `
    #     --num_warmup_updates $WarmupUpdates `
    #     --save_per_updates $SaveUpdates `
    #     --last_per_updates $LastUpdates `
    #     --finetune `
    #     --log_samples `
    #     --pretrain $PretrainCkpt

    if ($LASTEXITCODE -ne 0) {
        Write-Log "Fine-tuning failed!" "ERROR"
        exit 1
    }
    Write-Log "Stage 5 completed."
}

Write-Host ""
Write-Log "Fine-tuning pipeline completed successfully!" "SUCCESS"
