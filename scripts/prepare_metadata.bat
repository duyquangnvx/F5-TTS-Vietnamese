@echo off
REM Prepare metadata for F5-TTS training dataset
REM Usage: prepare_metadata.bat <dataset_dir> <training_dir> [options]

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: prepare_metadata.bat ^<dataset_dir^> ^<training_dir^> [--min-duration ^<sec^>] [--max-duration ^<sec^>] [--min-words ^<num^>]
    echo.
    echo Example: prepare_metadata.bat data\raw_audio data\training
    exit /b 1
)

python -m f5_tts.tools.prepare_metadata --dataset-dir "%~1" --training-dir "%~2" %3 %4 %5 %6 %7 %8 %9

endlocal
