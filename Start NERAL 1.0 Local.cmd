@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Neral-1.0-Local.ps1"
if errorlevel 1 pause
