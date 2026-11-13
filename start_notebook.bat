@echo off
cd /d "%~dp0"
set JUPYTER_PATH=%~dp0.venv\share\jupyter
jupyter lab project01_05_15_haih.ipynb
