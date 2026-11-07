@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-NERAL-Mini.ps1"
if errorlevel 1 (
  echo.
  echo NERAL Mini konnte nicht gestartet werden. Lies die Fehlermeldung oben.
  pause
)
