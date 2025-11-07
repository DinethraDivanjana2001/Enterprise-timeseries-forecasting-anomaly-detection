#!/bin/bash
# Setup script for Linux/Mac - Creates virtual environment and installs dependencies

echo "======================================================================"
echo "TIME-SERIES FORECASTING PROJECT SETUP (Linux/Mac)"
echo "======================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Virtual environment created successfully!"

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Step 4: Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "======================================================================"
echo "SETUP COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Virtual environment created at: venv/"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  source venv/bin/activate"
echo "  python scripts/run_pipeline.py"
echo ""
echo "For more information, see README.md"
echo "======================================================================"
