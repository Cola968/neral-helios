@echo off
setlocal
cd /d "%~dp0"
REM Ausdrueckliches Opt-in: vorhandenes Qwen3.5:9B mit gemessenem 2048-Token-Profil.
REM Kein Download, keine Cloud, kein Training. Fehlt Qwen3.5, endet der Start mit Fehlermeldung.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Neral-1.0-Local.ps1" -Model qwen3.5:9b -ExperimentalSampling
if errorlevel 1 pause
