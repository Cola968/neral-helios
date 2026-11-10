@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_helios_knowledge.ps1"
if errorlevel 1 (
  echo Helios Knowledge was not started. Review the error above; no training or fallback is triggered.
  pause
  exit /b 1
)
pause
