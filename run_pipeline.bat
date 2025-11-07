@echo off
REM Run the complete pipeline in virtual environment (Windows)

echo ======================================================================
echo RUNNING TIME-SERIES FORECASTING PIPELINE
echo ======================================================================

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_venv.bat first to create the virtual environment.
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running pipeline...
python scripts\run_pipeline.py

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline execution failed!
    exit /b 1
)

echo.
echo ======================================================================
echo PIPELINE COMPLETED!
echo ======================================================================
echo.
echo Check the following directories for outputs:
echo   - models\       (trained models)
echo   - outputs\      (predictions and anomalies)
echo   - reports\      (metrics and visualizations)
echo ======================================================================

pause
