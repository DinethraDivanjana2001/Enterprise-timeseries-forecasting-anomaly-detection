@echo off
REM Setup script for Windows - Creates virtual environment and installs dependencies

echo ======================================================================
echo TIME-SERIES FORECASTING PROJECT SETUP (Windows)
echo ======================================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

echo.
echo Step 1: Creating virtual environment...
python -m venv venv

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

echo Virtual environment created successfully!

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo ======================================================================
echo SETUP COMPLETED SUCCESSFULLY!
echo ======================================================================
echo.
echo Virtual environment created at: venv\
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To run the pipeline:
echo   venv\Scripts\activate
echo   python scripts\run_pipeline.py
echo.
echo For more information, see README.md
echo ======================================================================

pause
