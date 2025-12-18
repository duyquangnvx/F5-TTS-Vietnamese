@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============================================================================
:: F5-TTS Batch Evaluation Script for Windows
:: =============================================================================

:: Get script directory and set PYTHONPATH
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%
cd /d "%PROJECT_ROOT%"

:: Output Configuration
set OUTPUT_DIR=output\eval
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Evaluation Parameters
set MODEL=F5TTS_v1_Base
set NFE_STEP=16
set GPU_NUMS=1

:: =============================================================================
:: Batch Inference Evaluation
:: =============================================================================
echo [INFO] Starting batch inference evaluation...
echo [INFO] Model: %MODEL%
echo [INFO] NFE Steps: %NFE_STEP%
echo.

:: F5-TTS v1 Base - Chinese Test Set
echo [INFO] Evaluating on seedtts_test_zh...
accelerate launch src\f5_tts\eval\eval_infer_batch.py ^
    -s 0 -n "%MODEL%" -t "seedtts_test_zh" -nfe %NFE_STEP%

:: F5-TTS v1 Base - English Test Set
echo [INFO] Evaluating on seedtts_test_en...
accelerate launch src\f5_tts\eval\eval_infer_batch.py ^
    -s 0 -n "%MODEL%" -t "seedtts_test_en" -nfe %NFE_STEP%

:: F5-TTS v1 Base - LibriSpeech Test Clean
echo [INFO] Evaluating on ls_pc_test_clean...
accelerate launch src\f5_tts\eval\eval_infer_batch.py ^
    -s 0 -n "%MODEL%" -t "ls_pc_test_clean" -nfe %NFE_STEP%

:: =============================================================================
:: Metrics Evaluation
:: =============================================================================
set RESULTS_DIR=output\results\%MODEL%_1250000\seedtts_test_zh\seed0_euler_nfe%NFE_STEP%_vocos_ss-1_cfg2.0_speed1.0

echo.
echo [INFO] Computing WER metrics...
python src\f5_tts\eval\eval_seedtts_testset.py ^
    -e wer -l zh ^
    --gen_wav_dir "%RESULTS_DIR%" ^
    --gpu_nums %GPU_NUMS%

echo [INFO] Computing similarity metrics...
python src\f5_tts\eval\eval_seedtts_testset.py ^
    -e sim -l zh ^
    --gen_wav_dir "%RESULTS_DIR%" ^
    --gpu_nums %GPU_NUMS%

echo [INFO] Computing UTMOS metrics...
python src\f5_tts\eval\eval_utmos.py ^
    --audio_dir "%RESULTS_DIR%"

echo.
echo [SUCCESS] Evaluation completed!
echo [INFO] Results saved to: %OUTPUT_DIR%

endlocal
