@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if /I "%~1"=="--recreate" (
  if exist ".venv" rmdir /s /q ".venv"
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pip --version >nul 2>nul
  if not errorlevel 1 goto install_packages
  echo Removing an incomplete .venv from a previous setup...
  rmdir /s /q ".venv"
)

echo [1/4] Looking for Python 3.11-3.13...
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 goto create_with_py312
  py -3.13 -c "import sys" >nul 2>nul
  if not errorlevel 1 goto create_with_py313
  py -3.11 -c "import sys" >nul 2>nul
  if not errorlevel 1 goto create_with_py311
)

where python >nul 2>nul
if errorlevel 1 goto python_missing
python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 goto python_version
python -m venv ".venv"
if errorlevel 1 goto setup_failed
goto venv_created

:create_with_py312
py -3.12 -m venv ".venv"
if errorlevel 1 goto setup_failed
goto venv_created

:create_with_py313
py -3.13 -m venv ".venv"
if errorlevel 1 goto setup_failed
goto venv_created

:create_with_py311
py -3.11 -m venv ".venv"
if errorlevel 1 goto setup_failed
goto venv_created

:venv_created
if not exist ".venv\Scripts\python.exe" goto setup_failed

:install_packages
echo [2/4] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed

echo [3/4] Installing FLUID-Space packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

echo [4/4] Checking the application...
".venv\Scripts\python.exe" app.py --check
if errorlevel 1 goto setup_failed

echo.
echo Environment check passed.
echo Start the application with start_ui.bat
echo.
if /I not "%FLUID_SPACE_NO_PAUSE%"=="1" pause
exit /b 0

:python_missing
echo.
echo Python was not found. Install Python 3.12 (64-bit) from:
echo https://www.python.org/downloads/windows/
echo Then close this window and run setup_env.bat again.
if /I not "%FLUID_SPACE_NO_PAUSE%"=="1" pause
exit /b 1

:python_version
echo.
echo FLUID-Space requires Python 3.11, 3.12, or 3.13.
echo Python 3.12 (64-bit) is recommended.
if /I not "%FLUID_SPACE_NO_PAUSE%"=="1" pause
exit /b 1

:setup_failed
echo.
echo Setup failed. Review the error above, then run setup_env.bat again.
if /I not "%FLUID_SPACE_NO_PAUSE%"=="1" pause
exit /b 1
