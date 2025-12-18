@echo off
REM Convert audio sample rate to 24kHz for F5-TTS training
REM Usage: convert_sr.bat <input_dir> [options]

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: convert_sr.bat ^<input_dir^> [--pattern ^<glob^>] [--sample-rate ^<hz^>] [--workers ^<num^>] [--keep-original] [--in-place]
    echo.
    echo Example: convert_sr.bat data\wavs --in-place
    echo.
    echo Note: Requires sox to be installed. Download from: https://sourceforge.net/projects/sox/
    exit /b 1
)

python -m f5_tts.tools.convert_sr --input-dir "%~1" %2 %3 %4 %5 %6 %7 %8 %9

endlocal
