#!/usr/bin/env python3
"""
Check if the necessary dependencies are installed.
"""
import importlib
import subprocess
import sys
from pathlib import Path

# List of required packages
REQUIRED_PACKAGES = [
    "opencv-python",
    "numpy",
    "tqdm",
    "pillow",
    "scipy",
    "splitfolders"
]

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name.replace("-", "_").split(">=")[0])
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    project_root = Path(__file__).resolve().parents[2]
    
    print("Checking for required dependencies...")
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        package_name = package.split(">=")[0]  # Remove version specifier if any
        if not check_package(package_name):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"The following packages are missing: {', '.join(missing_packages)}")
        choice = input("Would you like to install them now? (y/n): ")
        
        if choice.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"Successfully installed {package}")
                else:
                    print(f"Failed to install {package}")
        else:
            print("\nPlease install the missing packages manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("All required dependencies are installed!")
    
    # Check if the FoHIS module files exist
    fohis_dir = project_root / "models" / "FoHIS-master" / "FoHIS"
    if not fohis_dir.exists():
        print(f"Warning: The FoHIS module directory does not exist at {fohis_dir}")
        print("This is okay - we'll use our built-in fog implementation instead of the FoHIS one.")
    else:
        required_files = ["fog.py", "parameter.py", "tool_kit.py", "const.py"]
        missing_files = [f for f in required_files if not (fohis_dir / f).exists()]
        
        if missing_files:
            print(f"Warning: The following files are missing from the FoHIS module: {', '.join(missing_files)}")
            print("This is okay - we'll use our built-in fog implementation instead of the FoHIS one.")
        else:
            print("FoHIS module files found.")
    
    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 