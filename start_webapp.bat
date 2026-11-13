@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Dang nap mo hinh, trinh duyet se tu mo khi san sang...
".venv\Scripts\python.exe" webapp\server.py --open
pause
