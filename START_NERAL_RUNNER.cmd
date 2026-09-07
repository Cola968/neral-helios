@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_NERAL_RUNNER.ps1"
echo.
if errorlevel 1 (
  echo Neral runner launcher needs attention. See the message above.
) else (
  echo Neral runner launcher finished successfully.
)
pause
