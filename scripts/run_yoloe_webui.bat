@echo off
setlocal
cd /d "%~dp0.."
call "%CD%\scripts\run_vision.bat" --webui %*
exit /b %ERRORLEVEL%
