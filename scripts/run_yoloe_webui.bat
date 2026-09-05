@echo off
setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
set "VISION_PYTHON="
if exist "%ROOT%\.venv\python.exe" (
    set "VISION_PYTHON=%ROOT%\.venv\python.exe"
    set "PATH=%ROOT%\.venv;%ROOT%\.venv\Library\bin;%PATH%"
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "VISION_PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    call "%ROOT%\isaac_env.bat" 2>nul
)
if not defined VISION_PYTHON set "VISION_PYTHON=%ISAAC_PYTHON%"
if not defined VISION_PYTHON set "VISION_PYTHON=python"
set "HF_HOME=%ROOT%\.cache\huggingface"
call "%VISION_PYTHON%" "%ROOT%\vision_main.py" --webui %*
exit /b %errorlevel%
