@echo off
rem Use this checkout's Isaac Sim 6.0.1 install (Python 3.12).
rem Do not use D:\isaac\env_isaaclab (that is 5.1 + Isaac Lab).
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "ISAAC_ROOT=%ROOT%\isaac-sim-6.0.1"
set "ISAAC_ENV=%ISAAC_ROOT%"
set "ISAAC_PYTHON=%ISAAC_ROOT%\python.bat"
set "OMNI_KIT_ACCEPT_EULA=YES"
if not exist "%ISAAC_PYTHON%" (
    echo Isaac Sim 6.0 Python launcher not found: "%ISAAC_PYTHON%"
    echo Install Isaac Sim 6.0.1 or update ISAAC_ROOT in isaac_env.bat.
    exit /b 1
)
