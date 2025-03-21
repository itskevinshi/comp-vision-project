#!/usr/bin/env python3
"""
Main entry point for the plant pathology detection project.
"""
import argparse
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Plant Pathology Detection with Weather Augmentation"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fog augmentation command
    fog_parser = subparsers.add_parser("fog", help="Apply fog augmentation to the dataset")
    fog_parser.add_argument(
        "--intensity", 
        type=str, 
        choices=["light", "medium", "heavy"], 
        default="medium",
        help="Intensity of the fog effect (default: medium)"
    )
    
    # Fog subset augmentation command
    fog_subset_parser = subparsers.add_parser("fog-subset", help="Apply fog augmentation to a subset of the dataset")
    fog_subset_parser.add_argument(
        "--intensity", 
        type=str, 
        choices=["light", "medium", "heavy"], 
        default="light",
        help="Intensity of the fog effect (default: light)"
    )
    fog_subset_parser.add_argument(
        "--percentage", 
        type=int, 
        default=10,
        help="Percentage of images to augment (default: 10)"
    )
    
    # Check dependencies command
    subparsers.add_parser("check-deps", help="Check if dependencies are installed")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "fog":
        # Check if FoHIS module is available
        fohis_dir = project_root / "models" / "FoHIS-master" / "FoHIS"
        using_builtin = not (fohis_dir.exists() and all(
            os.path.exists(fohis_dir / f) for f in ["fog.py", "parameter.py", "tool_kit.py", "const.py"]
        ))
        
        if using_builtin:
            print("Notice: Using built-in fog implementation instead of FoHIS.")
            print("This implementation is slower but doesn't require C++ build tools.")
            print("Computation time may be longer, especially for large images.")
        
        from src.weather_filters.apply_fog import main as apply_fog_main
        print(f"Applying {args.intensity} fog augmentation to the dataset...")
        apply_fog_main(args.intensity)
    
    elif args.command == "fog-subset":
        # Check if FoHIS module is available
        fohis_dir = project_root / "models" / "FoHIS-master" / "FoHIS"
        using_builtin = not (fohis_dir.exists() and all(
            os.path.exists(fohis_dir / f) for f in ["fog.py", "parameter.py", "tool_kit.py", "const.py"]
        ))
        
        if using_builtin:
            print("Notice: Using built-in fog implementation instead of FoHIS.")
            print("This implementation is slower but doesn't require C++ build tools.")
            print("Computation time may be longer, especially for large images.")
        
        from src.weather_filters.apply_fog_subset import main as apply_fog_subset_main
        print(f"Applying {args.intensity} fog augmentation to {args.percentage}% of the dataset...")
        apply_fog_subset_main(args.intensity, args.percentage)
        
    elif args.command == "check-deps":
        from src.utils.check_dependencies import main as check_deps_main
        check_deps_main()
        
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 