@echo off
setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
call "%ROOT%\isaac_env.bat" 2>nul
if not defined ISAAC_PYTHON set "ISAAC_PYTHON=python"
"%ISAAC_PYTHON%" "%ROOT%\vision_main.py" --webui %*
