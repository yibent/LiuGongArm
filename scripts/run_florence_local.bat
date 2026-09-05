@echo off
setlocal
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "FAST_BACKEND=yoloe"
if not exist "%~dp0..\yoloe-26x-seg.pt" set "FAST_BACKEND=cv"
if exist "%~dp0..\_models\yoloe\yoloe-26x-seg.pt" set "FAST_BACKEND=yoloe"
if defined BUSAGENT_YOLOE_WEIGHTS if exist "%BUSAGENT_YOLOE_WEIGHTS%" set "FAST_BACKEND=yoloe"
call "%~dp0run_yoloe_webui.bat" --fast %FAST_BACKEND% --prompt bus %*
exit /b %errorlevel%
