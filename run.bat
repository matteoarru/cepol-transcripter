@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe"
set "MAIN_SCRIPT=%SCRIPT_DIR%main.py"

if not exist "%PYTHON_EXE%" (
    echo ERROR: Virtual environment not found at "%SCRIPT_DIR%venv".
    echo Run setup first and then try again.
    exit /b 1
)

if "%~1"=="" (
    echo Usage: run.bat ROOT [options]
    echo Example: run.bat sample_media --log-level INFO
    echo Example: run.bat --version
    exit /b 1
)

"%PYTHON_EXE%" "%MAIN_SCRIPT%" %*
exit /b %ERRORLEVEL%
