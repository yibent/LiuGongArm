@echo off
rem Local Isaac Sim 6.0.1 (Python 3.12). Do not use D:\isaac\env_isaaclab (that is 5.1 + Isaac Lab).
set "ISAAC_ROOT=D:\isaacsim"
set "ISAAC_ENV=D:\isaacsim"
set "ISAAC_PYTHON=%ISAAC_ROOT%\python.bat"
set "OMNI_KIT_ACCEPT_EULA=YES"
if not exist "%ISAAC_PYTHON%" (
    echo Isaac Sim 6.0 Python launcher not found: "%ISAAC_PYTHON%"
    echo Install Isaac Sim 6.0.1 or update ISAAC_ROOT in isaac_env.bat.
    exit /b 1
)
