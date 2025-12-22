@echo off
REM ==========================================================
REM PyTorch Setup Script for F5-TTS
REM Automatically detects GPU and installs appropriate PyTorch
REM ==========================================================

REM Forward all arguments to PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0setup_pytorch.ps1" %*
