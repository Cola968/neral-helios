@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup-Arctus350.ps1"
if errorlevel 1 (
  echo.
  echo Arctus setup or qualification failed. See the message above.
  pause
  exit /b 1
)
