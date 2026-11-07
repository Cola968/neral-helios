@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-NERAL-Mini-Mobile.ps1"
if errorlevel 1 (
  echo.
  echo Fehler beim Start von NERAL Mini Mobile. Details oben lesen.
  pause
)
