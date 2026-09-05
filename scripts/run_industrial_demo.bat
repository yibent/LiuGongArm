@echo off
setlocal
cd /d "%~dp0.."
call isaac_env.bat || exit /b 1
"%ISAAC_PYTHON%" scripts\run_vision_follow.py --industrial-demo --no-follow %*
