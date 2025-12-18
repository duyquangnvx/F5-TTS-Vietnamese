@echo off
REM Check vocabulary coverage and extend with missing tokens
REM Usage: check_vocab.bat <pretrained_vocab> <dataset_vocab> <output> [--check-only]

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: check_vocab.bat ^<pretrained_vocab^> ^<dataset_vocab^> ^<output^> [--check-only]
    echo.
    echo Example: check_vocab.bat data\Emilia_ZH_EN_pinyin\vocab.txt data\my_dataset\vocab.txt data\my_dataset\vocab_extended.txt
    echo.
    echo Options:
    echo   --check-only    Only check for missing tokens without creating output
    exit /b 1
)

python -m f5_tts.tools.check_vocab --pretrained-vocab "%~1" --dataset-vocab "%~2" --output "%~3" %4 %5 %6 %7 %8 %9

endlocal
