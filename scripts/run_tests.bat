@echo off
setlocal
cd /d "%~dp0.."
set "ROOT=%CD%"
if exist "%ROOT%\isaac_env.bat" call "%ROOT%\isaac_env.bat"
if defined ISAAC_PYTHON (
    "%ISAAC_PYTHON%" -m unittest discover -s "%ROOT%\tests" -v
) else (
    python -m unittest discover -s "%ROOT%\tests" -v
)
