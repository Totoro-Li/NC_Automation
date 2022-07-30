@echo off
@cd /d %~dp0
if exist "venv" (
    echo Python environment is ready.
) else (
    echo Create virtual environment.
    python -m venv venv\ark-script)
call venv\Scripts\activate.bat
echo Install dependent libraries.
pip install -r requirements.txt
echo Finished.

pause