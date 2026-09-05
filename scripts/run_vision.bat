@echo off
setlocal
cd /d "%~dp0.."
if not defined VISION_PYTHON set "VISION_PYTHON=%CD%\_envs\vision\Scripts\python.exe"
if not exist "%VISION_PYTHON%" if exist "%CD%\.venv\python.exe" (
    set "VISION_PYTHON=%CD%\.venv\python.exe"
    set "PATH=%CD%\.venv;%CD%\.venv\Library\bin;%PATH%"
)
if not exist "%VISION_PYTHON%" if exist "%CD%\.venv\Scripts\python.exe" set "VISION_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%VISION_PYTHON%" (
    echo Run powershell -ExecutionPolicy Bypass -File scripts\setup_vision.ps1 first.
    exit /b 1
)
set "HF_HOME=%CD%\_models\hf_cache"
set "HF_HUB_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "YOLO_AUTOINSTALL=false"
set "YOLO_CONFIG_DIR=%CD%\.cache\ultralytics"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
"%VISION_PYTHON%" "%CD%\vision_main.py" %*
exit /b %ERRORLEVEL%
