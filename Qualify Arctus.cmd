@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Qualify-Arctus.ps1" %*
if errorlevel 1 (
  echo.
  echo Arctus qualification failed. See the message above.
  pause
  exit /b 1
)
echo.
pause
