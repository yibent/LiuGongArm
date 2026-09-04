@echo off
setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
call "%ROOT%\isaac_env.bat" || exit /b 1
"%ISAAC_PYTHON%" "%ROOT%\scripts\run_vision_follow.py" %*
