@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" make_release.py
) else (
  python make_release.py
)

if errorlevel 1 (
  echo Release creation failed.
  pause
  exit /b 1
)
pause
