@echo off
@cd /d %~dp0
call venv\Scripts\activate.bat
python main.py
taskkill /f /im adb.exe /t
pause