@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_helios_guarded.ps1"
if errorlevel 1 (
  echo Helios was not started. Review the error above; no fallback or training is triggered.
  pause
  exit /b 1
)
pause
