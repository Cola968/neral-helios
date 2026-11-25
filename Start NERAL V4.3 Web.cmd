@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0;%~dp0local_state\three-tier-web-deps"
set "PYTHONDONTWRITEBYTECODE=1"
"D:\neral-qlora\venv\Scripts\python.exe" -B "%~dp0neral_v43_verified_start.py"
pause
