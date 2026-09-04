@echo off
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
call "%ROOT%\isaac_env.bat" || exit /b 1
"%ISAAC_PYTHON%" "%ROOT%\scripts\hello_world.py" %*
