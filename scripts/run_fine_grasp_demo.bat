@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_fine_grasp_demo.ps1" %*
exit /b %ERRORLEVEL%
