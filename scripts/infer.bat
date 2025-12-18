@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============================================================================
:: F5-TTS Inference Script for Windows
:: =============================================================================

:: Get script directory and set PYTHONPATH
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%
cd /d "%PROJECT_ROOT%"

:: Output Configuration
set OUTPUT_DIR=output\inference
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Model Configuration
set MODEL=F5TTS_Base
set VOCAB_FILE=data\ViVoice\vocab.txt
set CKPT_FILE=data\ViVoice\model_last.pt
set VOCODER=vocos

:: Inference Parameters
set SPEED=1.0
set NFE_STEP=32

:: Input Files
set REF_AUDIO=ref\vi_ref_1.wav
set REF_TEXT=cả hai bên hãy cố gắng hiểu cho nhau
set GEN_TEXT=mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn, sau đó mang những gì học được về việt nam giúp xây dựng các công trình tốt hơn

:: Output File
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set OUTPUT_FILE=%OUTPUT_DIR%\output_%TIMESTAMP%.wav

:: =============================================================================
:: Run Inference
:: =============================================================================
echo [INFO] Running F5-TTS inference...
echo [INFO] Model: %MODEL%
echo [INFO] Checkpoint: %CKPT_FILE%
echo [INFO] Output: %OUTPUT_FILE%
echo.

python -m f5_tts.infer.infer_cli ^
    --model "%MODEL%" ^
    --ref_audio "%REF_AUDIO%" ^
    --ref_text "%REF_TEXT%" ^
    --gen_text "%GEN_TEXT%" ^
    --speed %SPEED% ^
    --nfe_step %NFE_STEP% ^
    --vocoder_name %VOCODER% ^
    --vocab_file "%VOCAB_FILE%" ^
    --ckpt_file "%CKPT_FILE%" ^
    --output_dir "%OUTPUT_DIR%"

if errorlevel 1 (
    echo [ERROR] Inference failed!
    exit /b 1
)

echo.
echo [SUCCESS] Inference completed!
echo [INFO] Output saved to: %OUTPUT_DIR%

endlocal
