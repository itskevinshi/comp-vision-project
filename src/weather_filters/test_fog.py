#!/usr/bin/env python3
"""
Test script for the fog augmentation.
This script applies fog to a single image to verify that the implementation works.
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import our fog implementation
from src.weather_filters.apply_fog import apply_fog_effect
from src.weather_filters.perlin import generate_perlin_noise

def main():
    # Create a test directory if it doesn't exist
    test_dir = project_root / "outputs" / "test_fog"
    os.makedirs(test_dir, exist_ok=True)
    
    # Try to find an image from the processed dataset
    processed_dir = project_root / "data" / "processed"
    
    # Find the first image in the processed dataset
    image_path = None
    for split in ['train', 'val', 'test']:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue
            
        for class_dir in os.listdir(split_dir):
            class_path = split_dir / class_dir
            if not class_path.is_dir():
                continue
                
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = class_path / file
                    break
            if image_path:
                break
        if image_path:
            break
    
    # If no image was found, create a simple test image
    if image_path is None:
        print("No images found in the processed dataset. Creating a test image.")
        # Create a simple green "leaf" image
        test_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        # Make a green circle in the middle
        cv2.circle(test_image, (150, 150), 100, (0, 200, 0), -1)
        
        # Save the test image
        test_image_path = test_dir / "test_leaf.jpg"
        cv2.imwrite(str(test_image_path), test_image)
        image_path = test_image_path
    
    print(f"Testing fog effect on image: {image_path}")
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return 1
    
    # Apply fog effect with different intensities
    intensities = ['light', 'medium', 'heavy']
    
    for intensity in intensities:
        print(f"Applying {intensity} fog...")
        
        # Set fog parameters based on intensity
        if intensity == 'light':
            visibility = 24
            fog_top = 40
        elif intensity == 'medium':
            visibility = 12
            fog_top = 70
        elif intensity == 'heavy':
            visibility = 6
            fog_top = 100
        
        # Create a foggy version of the image
        foggy_image = apply_fog_effect(image)
        
        # Save the result
        output_path = test_dir / f"foggy_{intensity}_{os.path.basename(image_path)}"
        cv2.imwrite(str(output_path), foggy_image)
        print(f"Saved foggy image to {output_path}")
    
    # Also generate and save a perlin noise sample
    print("Generating Perlin noise sample...")
    noise = generate_perlin_noise(300, 300, scale=30.0, octaves=6)
    noise_image = (noise * 255).astype(np.uint8)
    noise_path = test_dir / "perlin_noise_sample.jpg"
    cv2.imwrite(str(noise_path), noise_image)
    print(f"Saved noise sample to {noise_path}")
    
    print("Fog test completed! Check the output images in", test_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main()) 