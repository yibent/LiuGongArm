@echo off
rem Local Isaac Sim 6.0.1 (Python 3.12). Do not use D:\isaac\env_isaaclab (that is 5.1 + Isaac Lab).
set "ISAAC_ROOT=D:\isaac"
set "ISAAC_ENV=D:\isaac\env_isaacsim60"
set "ISAAC_PYTHON=%ISAAC_ENV%\python.exe"
set "OMNI_KIT_ACCEPT_EULA=YES"
if not exist "%ISAAC_PYTHON%" (
    echo Isaac Sim 6.0 Python not found: "%ISAAC_PYTHON%"
    echo Create it with: conda create -y -p D:\isaac\env_isaacsim60 python=3.12
    echo Then: pip install isaacsim[all,extscache]==6.0.1.0 --extra-index-url https://pypi.nvidia.com
    exit /b 1
)
