@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo FLUID-Space is not installed yet. Starting setup...
  set "FLUID_SPACE_NO_PAUSE=1"
  call setup_env.bat
  if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" run_profile.py --profile current_baseline
if errorlevel 1 (
  echo.
  echo Pipeline stopped. Review the error above.
  pause
  exit /b 1
)

echo.
echo Result completed under profiles\current_baseline\runs
pause
