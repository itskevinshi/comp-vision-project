import os
import cv2
import numpy as np
import sys
import random
from pathlib import Path
from tqdm import tqdm

# Import our fog implementation
from src.weather_filters.apply_fog import apply_fog_effect

# Get project root
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to reach project root

def process_dataset_subset(input_dir, output_dir, fog_intensity='medium', subset_percentage=10):
    """
    Process a subset of images in the input directory and save foggy versions to output directory.
    
    Args:
        input_dir: Directory containing clean images
        output_dir: Directory to save foggy images
        fog_intensity: 'light', 'medium', or 'heavy'
        subset_percentage: Percentage of images to process (e.g., 10 for 10%)
    """
    # Import constants from apply_fog
    from src.weather_filters.apply_fog import const
    
    # Set fog parameters based on intensity
    if fog_intensity == 'light':
        const.VISIBILITY_RANGE_MOLECULE = 24
        const.FT = 40
    elif fog_intensity == 'medium':
        const.VISIBILITY_RANGE_MOLECULE = 12
        const.FT = 70
    elif fog_intensity == 'heavy':
        const.VISIBILITY_RANGE_MOLECULE = 6
        const.FT = 100
    
    # Update extinction coefficient
    const.ECM = 3.912 / const.VISIBILITY_RANGE_MOLECULE
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all subdirectories (class folders)
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Create output class directory
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        print(f"Processing {class_name}...")
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Warning: No images found in {class_name}, skipping...")
            continue
        
        # Calculate number of images to process based on subset_percentage
        num_to_process = max(1, int(len(image_files) * subset_percentage / 100))
        
        # If we have fewer images than the requested percentage, use all images
        if num_to_process > len(image_files):
            num_to_process = len(image_files)
            print(f"Note: Using all {num_to_process} images (less than requested {subset_percentage}%)")
        
        # Randomly select images to process
        selected_images = random.sample(image_files, num_to_process)
        
        print(f"Selected {num_to_process} out of {len(image_files)} images")
        
        # Process the selected images
        for image_file in tqdm(selected_images):
            # Read image
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read {image_path}")
                continue
            
            # Apply fog effect
            foggy_image = apply_fog_effect(image)
            
            # Save foggy image
            output_path = os.path.join(output_class_dir, image_file)
            cv2.imwrite(output_path, foggy_image)

def main(fog_intensity='medium', subset_percentage=10):
    """
    Main function to apply fog augmentation to a subset of the dataset.
    
    Args:
        fog_intensity: 'light', 'medium', or 'heavy'
        subset_percentage: Percentage of images to process (default: 10%)
    """
    # Define paths
    processed_dir = project_root / "data" / "processed"
    augmented_weather_dir = project_root / "data" / "augmented" / "weather" / "fog"
    
    # Create fog directories for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        input_dir = processed_dir / split
        output_dir = augmented_weather_dir / split
        
        print(f"\nProcessing {split} dataset with {fog_intensity} fog...")
        process_dataset_subset(str(input_dir), str(output_dir), 
                              fog_intensity=fog_intensity, 
                              subset_percentage=subset_percentage)
    
    print(f"\nFog augmentation completed on {subset_percentage}% of the dataset!")

if __name__ == "__main__":
    # If script is run directly, use light intensity and 10% subset by default
    main('light', 10) 