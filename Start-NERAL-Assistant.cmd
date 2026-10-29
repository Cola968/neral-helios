@echo off
setlocal
cd /d "%~dp0"
title NERAL Assistant - lokale KI starten

rem This entry point does NOT download models or Python packages.
rem The installed Ollama qwen3.5:9b model is required.
set "NERAL_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "NERAL_PYTHON=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" if exist "local_state\python\Scripts\python.exe" set "NERAL_PYTHON=local_state\python\Scripts\python.exe"

"%NERAL_PYTHON%" neral_assistant_launcher.py
if errorlevel 1 (
  echo.
  echo NERAL konnte noch nicht gestartet werden. Siehe Meldung oben.
  echo Lese docs\NERAL_QWEN_INTEGRATION.md fuer die Einrichtung.
  pause
  exit /b 1
)
echo.
echo NERAL Assistant laeuft im Browser unter http://127.0.0.1:9680
exit /b 0

