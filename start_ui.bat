@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  echo FLUID-Space is not installed yet. Starting setup...
  set "FLUID_SPACE_NO_PAUSE=1"
  call setup_env.bat
  if errorlevel 1 exit /b 1
)

start "FLUID-Space" ".venv\Scripts\pythonw.exe" app.py --profile current_baseline
exit /b 0
