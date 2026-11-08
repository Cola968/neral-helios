@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Neral-Local-Voice.ps1"
if errorlevel 1 echo Sprachinstallation fehlgeschlagen. Details bitte oben lesen.
pause
