@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0voicebank-benchmark.ps1"
echo Exit code: %ERRORLEVEL%
pause
