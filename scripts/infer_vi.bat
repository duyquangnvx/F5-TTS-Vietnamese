@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============================================================================
:: F5-TTS Vietnamese Inference Script (using config file)
:: Uses TOML config to avoid UTF-8 encoding issues with command line
:: =============================================================================

:: Set UTF-8 encoding for Python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: Get script directory and set PYTHONPATH
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%
cd /d "%PROJECT_ROOT%"

:: Config file (edit configs\infer_vi.toml to change text)
set CONFIG_FILE=configs\infer_vi.toml

echo ==============================================
echo F5-TTS Vietnamese Inference
echo ==============================================
echo Config: %CONFIG_FILE%
echo ==============================================
echo.

:: Run inference with config file
python -m f5_tts.infer.infer_cli --config "%CONFIG_FILE%"

if errorlevel 1 (
    echo [ERROR] Inference failed!
    exit /b 1
)

echo.
echo [SUCCESS] Inference completed!

endlocal
