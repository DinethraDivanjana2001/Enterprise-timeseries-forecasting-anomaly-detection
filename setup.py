"""
Setup script for the time-series forecasting project.

This script:
1. Checks Python version
2. Creates necessary directories
3. Installs dependencies
4. Verifies installation
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is 3.8 or higher."""
    logger.info("Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} - OK")
    return True


def create_directories():
    """Create necessary project directories."""
    logger.info("Creating project directories...")
    
    project_root = Path(__file__).parent
    directories = [
        "data",
        "models",
        "outputs",
        "reports",
        "notebooks"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"  Created/verified: {dir_name}/")
    
    logger.info("Directories created successfully")


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    logger.info("Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def verify_installation():
    """Verify that key packages are installed."""
    logger.info("Verifying installation...")
    
    required_packages = [
        "numpy",
        "pandas",
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "yaml"
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  {package} - OK")
        except ImportError:
            logger.error(f"  {package} - MISSING")
            all_installed = False
    
    return all_installed


def main():
    """Main setup function."""
    logger.info("="*70)
    logger.info("TIME-SERIES FORECASTING PROJECT SETUP")
    logger.info("="*70)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    logger.info("\n" + "="*70)
    logger.info("INSTALLING DEPENDENCIES")
    logger.info("="*70)
    
    if not install_dependencies():
        logger.error("Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    logger.info("\n" + "="*70)
    logger.info("VERIFYING INSTALLATION")
    logger.info("="*70)
    
    if not verify_installation():
        logger.error("Setup failed during verification")
        sys.exit(1)
    
    # Success
    logger.info("\n" + "="*70)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("  1. Review configuration files in configs/")
    logger.info("  2. Run the pipeline: python scripts/run_pipeline.py")
    logger.info("  3. Or run individual scripts in scripts/")
    logger.info("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
