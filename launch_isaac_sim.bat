@echo off
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
call "%ROOT%\isaac_env.bat" || exit /b 1

echo Starting Isaac Sim 6.0.1 with project extension and default scene...
"%ISAAC_PYTHON%" -m isaacsim isaacsim.exp.full ^
    --ext-folder "%ROOT%\extensions" ^
    --enable mr_liu.project ^
    "%ROOT%\scenes\world.usda" %*
