#!/usr/bin/env python3
"""
Run fog augmentation on the processed dataset
"""
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.weather_filters.apply_fog import main as apply_fog_main

if __name__ == "__main__":
    # Ensure the output directories exist
    weather_dir = project_root / "data" / "augmented" / "weather" / "fog"
    for split in ['train', 'val', 'test']:
        os.makedirs(weather_dir / split, exist_ok=True)
    
    # Run the fog augmentation
    print("Starting fog augmentation process...")
    apply_fog_main() 