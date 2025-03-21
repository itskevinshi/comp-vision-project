import sys
import argparse
from pathlib import Path

# Add the project root to the path for imports to work correctly
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to reach project root
sys.path.append(str(project_root))

# Import the raindrop augmentation function using CV implementation
from src.weather_filters.apply_raindrop_cv import main as apply_raindrop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply raindrop augmentation to datasets')
    parser.add_argument('--intensity', type=str, default='medium', choices=['light', 'medium', 'heavy'],
                        help='Intensity of raindrop effect (default: medium)')
    parser.add_argument('--percentage', type=int, default=10, 
                        help='Percentage of images to apply effect to (default: 10)')
    
    args = parser.parse_args()
    
    # Apply raindrop effects
    apply_raindrop(args.intensity, args.percentage) 