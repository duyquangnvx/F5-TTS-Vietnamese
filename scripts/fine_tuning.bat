@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============================================================================
:: F5-TTS Fine-tuning Pipeline for Windows
:: =============================================================================

:: Get script directory and set PYTHONPATH
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%
cd /d "%PROJECT_ROOT%"

:: GPU Configuration
set CUDA_VISIBLE_DEVICES=0

:: Directory Configuration
set DATASET_DIR=data\your_training_dataset
set OUTPUT_DIR=output
set RAW_DATASET_DIR=data\your_dataset

:: Training Parameters
set EXP_NAME=F5TTS_Base
set DATASET_NAME=your_training_dataset
set BATCH_SIZE=7000
set NUM_WORKERS=16
set WARMUP_UPDATES=20000
set SAVE_UPDATES=10000
set LAST_UPDATES=10000
set PRETRAIN_CKPT=ckpts\your_training_dataset\pretrained_model_1200000.pt

:: Pipeline Stage Control (set start and stop stage)
:: Stage 0: Convert sample rate
:: Stage 1: Prepare metadata
:: Stage 2: Check vocabulary
:: Stage 3: Extend embeddings
:: Stage 4: Feature extraction
:: Stage 5: Fine-tuning
set STAGE=5
set STOP_STAGE=5

:: =============================================================================
:: Create necessary directories
:: =============================================================================
echo [INFO] Creating directories...
if not exist "%DATASET_DIR%" mkdir "%DATASET_DIR%"
if not exist "%RAW_DATASET_DIR%" mkdir "%RAW_DATASET_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\logs" mkdir "%OUTPUT_DIR%\logs"

:: =============================================================================
:: Stage 0: Convert sample rate to 24kHz
:: =============================================================================
if %STAGE% LEQ 0 if %STOP_STAGE% GEQ 0 (
    echo [INFO] Stage 0: Converting sample rate to 24kHz...
    python -m f5_tts.tools.convert_sr ^
        --input-dir "%RAW_DATASET_DIR%" ^
        --pattern "*.wav" ^
        --sample-rate 24000 ^
        --workers %NUM_WORKERS% ^
        --keep-original
    if errorlevel 1 (
        echo [ERROR] Sample rate conversion failed!
        goto :error
    )
    echo [INFO] Stage 0 completed.
)

:: =============================================================================
:: Stage 1: Prepare metadata
:: =============================================================================
if %STAGE% LEQ 1 if %STOP_STAGE% GEQ 1 (
    echo [INFO] Stage 1: Preparing metadata...
    python -m f5_tts.tools.prepare_metadata ^
        --dataset-dir "%RAW_DATASET_DIR%" ^
        --training-dir "%DATASET_DIR%"
    if errorlevel 1 (
        echo [ERROR] Metadata preparation failed!
        goto :error
    )
    echo [INFO] Stage 1 completed.
)

:: =============================================================================
:: Stage 2: Check vocabulary
:: =============================================================================
if %STAGE% LEQ 2 if %STOP_STAGE% GEQ 2 (
    echo [INFO] Stage 2: Checking vocabulary...
    python -m f5_tts.tools.check_vocab ^
        --pretrained-vocab "data\Emilia_ZH_EN_pinyin\vocab.txt" ^
        --dataset-vocab "%DATASET_DIR%\vocab.txt" ^
        --output "%DATASET_DIR%\vocab_extended.txt"
    if errorlevel 1 (
        echo [ERROR] Vocabulary check failed!
        goto :error
    )
    echo [INFO] Stage 2 completed.
)

:: =============================================================================
:: Stage 3: Extend model embeddings
:: =============================================================================
if %STAGE% LEQ 3 if %STOP_STAGE% GEQ 3 (
    echo [INFO] Stage 3: Extending model embeddings...
    python -m f5_tts.tools.extend_embeddings ^
        --model F5TTS_Base ^
        --pretrained-vocab "data\Emilia_ZH_EN_pinyin\vocab.txt" ^
        --new-vocab "%DATASET_DIR%\vocab_extended.txt" ^
        --output "ckpts\%DATASET_NAME%\pretrained_model_1200000.pt"
    if errorlevel 1 (
        echo [ERROR] Embedding extension failed!
        goto :error
    )
    echo [INFO] Stage 3 completed.
)

:: =============================================================================
:: Stage 4: Feature extraction
:: =============================================================================
if %STAGE% LEQ 4 if %STOP_STAGE% GEQ 4 (
    echo [INFO] Stage 4: Extracting features...
    python src\f5_tts\train\datasets\prepare_csv_wavs.py ^
        "%DATASET_DIR%" "%DATASET_DIR%" ^
        --workers %NUM_WORKERS%
    if errorlevel 1 (
        echo [ERROR] Feature extraction failed!
        goto :error
    )
    echo [INFO] Stage 4 completed.
)

:: =============================================================================
:: Stage 5: Fine-tuning
:: =============================================================================
if %STAGE% LEQ 5 if %STOP_STAGE% GEQ 5 (
    echo [INFO] Stage 5: Starting fine-tuning...

    :: Single GPU training
    python src\f5_tts\train\finetune_cli.py ^
        --exp_name "%EXP_NAME%" ^
        --dataset_name "%DATASET_NAME%" ^
        --batch_size_per_gpu %BATCH_SIZE% ^
        --num_warmup_updates %WARMUP_UPDATES% ^
        --save_per_updates %SAVE_UPDATES% ^
        --last_per_updates %LAST_UPDATES% ^
        --finetune ^
        --log_samples ^
        --pretrain "%PRETRAIN_CKPT%"

    :: Multi-GPU training (uncomment to use)
    :: accelerate launch src\f5_tts\train\finetune_cli.py ^
    ::     --exp_name "%EXP_NAME%" ^
    ::     --dataset_name "%DATASET_NAME%" ^
    ::     --batch_size_per_gpu %BATCH_SIZE% ^
    ::     --num_warmup_updates %WARMUP_UPDATES% ^
    ::     --save_per_updates %SAVE_UPDATES% ^
    ::     --last_per_updates %LAST_UPDATES% ^
    ::     --finetune ^
    ::     --log_samples ^
    ::     --pretrain "%PRETRAIN_CKPT%"

    if errorlevel 1 (
        echo [ERROR] Fine-tuning failed!
        goto :error
    )
    echo [INFO] Stage 5 completed.
)

echo.
echo [SUCCESS] Fine-tuning pipeline completed successfully!
goto :end

:error
echo.
echo [FAILED] Pipeline failed at stage %STAGE%
exit /b 1

:end
endlocal
