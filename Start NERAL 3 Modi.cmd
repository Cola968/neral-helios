@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Neral-Three-Tier.ps1"
if errorlevel 1 pause
