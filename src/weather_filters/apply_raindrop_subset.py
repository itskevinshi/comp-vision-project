import os
import cv2
import numpy as np
import random
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add the project root to the path for imports to work correctly
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to reach project root
sys.path.append(str(project_root))

# Import raindrop generator function from ROLE-master
# Fix the import to avoid hyphen in module name
sys.path.append(str(project_root / "models" / "ROLE-master"))
from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg

def apply_raindrop_effect(image_path, drop_config=None):
    """
    Apply raindrop effect to an image.
    
    Args:
        image_path: Path to the image file
        drop_config: Configuration for raindrop effect (optional)
    
    Returns:
        PIL Image with raindrop effect applied
    """
    if drop_config is None:
        drop_config = cfg  # Use default config if none provided
        
    # Generate raindrops on the image
    output_image, _ = generateDrops(image_path, drop_config)
    
    return output_image

def process_dataset_subset(input_dir, output_dir, raindrop_intensity='medium', subset_percentage=10):
    """
    Process a subset of images in the input directory and save raindrop-augmented versions to output directory.
    
    Args:
        input_dir: Directory containing clean images
        output_dir: Directory to save raindrop-augmented images
        raindrop_intensity: 'light', 'medium', or 'heavy'
        subset_percentage: Percentage of images to process (e.g., 10 for 10%)
    """
    # Configure raindrop parameters based on intensity
    raindrop_config = cfg.copy()
    
    if raindrop_intensity == 'light':
        raindrop_config['maxDrops'] = 15
        raindrop_config['minDrops'] = 10
        raindrop_config['maxR'] = 30
        raindrop_config['minR'] = 20
    elif raindrop_intensity == 'medium':
        raindrop_config['maxDrops'] = 30
        raindrop_config['minDrops'] = 20
        raindrop_config['maxR'] = 40
        raindrop_config['minR'] = 25
    elif raindrop_intensity == 'heavy':
        raindrop_config['maxDrops'] = 45
        raindrop_config['minDrops'] = 30
        raindrop_config['maxR'] = 50
        raindrop_config['minR'] = 30
    
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
        #num_to_process = max(1, int(len(image_files) * subset_percentage / 100))
        num_to_process = len(image_files)
        
        # If we have fewer images than the requested percentage, use all images
        if num_to_process > len(image_files):
            num_to_process = len(image_files)
            print(f"Note: Using all {num_to_process} images (less than requested {subset_percentage}%)")
        
        # Randomly select images to process
        selected_images = random.sample(image_files, num_to_process)
        
        print(f"Selected {num_to_process} out of {len(image_files)} images")
        
        # Process the selected images
        for image_file in tqdm(selected_images):
            # Full path to the image
            image_path = os.path.join(class_dir, image_file)
            
            try:
                # Apply raindrop effect
                raindrop_image = apply_raindrop_effect(image_path, raindrop_config)
                
                # Save raindrop image
                output_path = os.path.join(output_class_dir, image_file)
                raindrop_image.save(output_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

def main(raindrop_intensity='medium', subset_percentage=10):
    """
    Main function to apply raindrop augmentation to a subset of the dataset.
    
    Args:
        raindrop_intensity: 'light', 'medium', or 'heavy'
        subset_percentage: Percentage of images to process (default: 10%)
    """
    # Define paths
    # processed_dir = project_root / "data" / "processed"
    # augmented_weather_dir = project_root / "data" / "augmented" / "weather" / "raindrop"

    processed_dir = project_root / "plant_village_limited_split" / "processed"
    augmented_weather_dir = project_root / "plant_village_limited_split" / "augmented" / "weather" / "raindrop"
    
    # Create raindrop directories for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        input_dir = processed_dir / split
        output_dir = augmented_weather_dir / split
        
        print(f"\nProcessing {split} dataset with {raindrop_intensity} raindrop effect...")
        process_dataset_subset(str(input_dir), str(output_dir), 
                             raindrop_intensity=raindrop_intensity, 
                             subset_percentage=subset_percentage)
    
    print(f"\nRaindrop augmentation completed on {subset_percentage}% of the dataset!")

if __name__ == "__main__":
    # If script is run directly, use medium intensity and 10% subset by default
    main('medium', 10) 