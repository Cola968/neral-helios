@echo off
setlocal
cd /d "%~dp0"
echo NERAL CYBER GUARD - lokaler Windows-Sicherheitscheck
echo Keine Einstellungen werden geaendert. Keine Cloud.
echo.
set "NERAL_CYBER_PYTHON=%USERPROFILE%\NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe"
if exist "%NERAL_CYBER_PYTHON%" (
    "%NERAL_CYBER_PYTHON%" -B -m cyber.guard
    goto done
)
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -B -m cyber.guard
    goto done
)
where python >nul 2>&1
if not errorlevel 1 (
    python -B -m cyber.guard
    goto done
)
echo Python nicht gefunden. Bitte Python 3 installieren oder NERAL-Umgebung pruefen.
:done
echo.
pause
