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

def create_raindrop_mask(height, width, center_x, center_y, radius, oval_factor=1.3):
    """
    Create a raindrop mask with a circle and an oval to simulate a raindrop shape.
    
    Args:
        height, width: Dimensions of the mask
        center_x, center_y: Center coordinates of the drop
        radius: Radius of the drop
        oval_factor: Factor for oval height compared to radius
        
    Returns:
        Mask of the raindrop
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw circle
    cv2.circle(mask, (center_x, center_y), radius, 128, -1)
    
    # Draw oval (ellipse) for the bottom part
    oval_height = int(oval_factor * radius)
    cv2.ellipse(mask, (center_x, center_y), (radius, oval_height), 0, 180, 360, 128, -1)
    
    # Blur the mask to create smooth edges
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    
    # Normalize the mask
    mask = mask.astype(np.float32) / np.max(mask) * 255.0
    
    return mask

def create_raindrop_texture(image, center_x, center_y, radius, mask):
    """
    Create a raindrop texture by applying a fisheye-like effect on a portion of the image.
    
    Args:
        image: Input image
        center_x, center_y: Center coordinates of the drop
        radius: Radius of the drop
        mask: Mask of the raindrop
        
    Returns:
        Raindrop texture
    """
    # Extract region of interest (ROI)
    roi_size = radius * 4
    roi_x1 = max(0, center_x - roi_size//2)
    roi_x2 = min(image.shape[1], center_x + roi_size//2)
    roi_y1 = max(0, center_y - roi_size//2)
    roi_y2 = min(image.shape[0], center_y + roi_size//2)
    
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    
    if roi.size == 0:
        return None
    
    # Apply Gaussian blur to simulate refraction
    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 2)
    
    # Create a distortion map (simple version)
    roi_h, roi_w = roi.shape[:2]
    map_x, map_y = np.meshgrid(np.linspace(0, roi_w-1, roi_w), np.linspace(0, roi_h-1, roi_h))
    
    # Calculate distances from center of ROI
    center_roi_x, center_roi_y = roi_w // 2, roi_h // 2
    dx = map_x - center_roi_x
    dy = map_y - center_roi_y
    
    # Calculate normalized distance
    r = np.sqrt(dx**2 + dy**2)
    r = r / (np.max(r) * 0.8)  # Scale factor
    
    # Apply distortion (simple lens effect)
    k = 0.3  # Distortion factor
    r_dist = r * (1 + k * r**2)
    
    # Clamp r_dist to avoid overflow
    r_dist = np.minimum(r_dist, 1.0)
    
    # Calculate new coordinates
    theta = np.arctan2(dy, dx)
    new_x = center_roi_x + r_dist * np.cos(theta) * r * roi_w/2
    new_y = center_roi_y + r_dist * np.sin(theta) * r * roi_h/2
    
    # Ensure coordinates are within bounds
    new_x = np.clip(new_x, 0, roi_w-1).astype(np.float32)
    new_y = np.clip(new_y, 0, roi_h-1).astype(np.float32)
    
    # Remap image
    distorted_roi = cv2.remap(blurred_roi, new_x, new_y, cv2.INTER_LINEAR)
    
    # Flip the ROI vertically to simulate water drop reflection
    distorted_roi = cv2.flip(distorted_roi, 0)
    
    return distorted_roi

def add_raindrop(image, config):
    """
    Add raindrops to an image.
    
    Args:
        image: Input image
        config: Configuration parameters for raindrops
        
    Returns:
        Image with raindrops
    """
    img_height, img_width = image.shape[:2]
    result = image.copy()
    
    # Number of raindrops
    drop_count = random.randint(config['minDrops'], config['maxDrops'])
    
    # Track occupied areas to avoid overlapping drops
    occupied = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for _ in range(drop_count):
        # Random raindrop parameters
        radius = random.randint(config['minR'], config['maxR'])
        center_x = random.randint(radius, img_width - radius)
        center_y = random.randint(radius, img_height - radius)
        
        # Check if the center area is already occupied
        if occupied[center_y, center_x] > 0:
            continue
        
        # Create a mask for the drop (circle + oval)
        drop_mask_height = 5 * radius
        drop_mask_width = 4 * radius
        drop_mask = create_raindrop_mask(drop_mask_height, drop_mask_width, 
                                        drop_mask_width // 2, drop_mask_height // 2, radius)
        
        # Adjust mask to image coordinates
        mask_y1 = max(0, center_y - drop_mask_height // 2)
        mask_y2 = min(img_height, center_y + drop_mask_height // 2)
        mask_x1 = max(0, center_x - drop_mask_width // 2)
        mask_x2 = min(img_width, center_x + drop_mask_width // 2)
        
        mask_h = mask_y2 - mask_y1
        mask_w = mask_x2 - mask_x1
        
        # Check if mask dimensions are valid
        if mask_h <= 0 or mask_w <= 0:
            continue
            
        # Get portion of the mask that fits within the image
        mask_offset_y = max(0, drop_mask_height // 2 - center_y)
        mask_offset_x = max(0, drop_mask_width // 2 - center_x)
        crop_mask = drop_mask[mask_offset_y:mask_offset_y + mask_h, 
                              mask_offset_x:mask_offset_x + mask_w]
        
        # Create a raindrop texture
        drop_texture = create_raindrop_texture(image, center_x, center_y, radius, crop_mask)
        
        if drop_texture is None or drop_texture.size == 0:
            continue
        
        # Resize texture to match the mask region
        try:
            drop_texture = cv2.resize(drop_texture, (mask_w, mask_h))
        except cv2.error:
            continue
            
        # Convert mask to 3 channels for color image blending
        mask_3ch = np.repeat(crop_mask[:, :, np.newaxis] / 255.0, 3, axis=2)
        
        # Blend the original image with the drop texture
        roi = result[mask_y1:mask_y2, mask_x1:mask_x2]
        result[mask_y1:mask_y2, mask_x1:mask_x2] = (
            drop_texture * mask_3ch + roi * (1 - mask_3ch)
        ).astype(np.uint8)
        
        # Mark the area as occupied
        occupied[mask_y1:mask_y2, mask_x1:mask_x2] = (crop_mask > 0).astype(np.uint8)
    
    # Apply darken effect at edges for more realism
    if config.get('edge_darkratio', 0) > 0:
        edge_mask = occupied.astype(np.float32) / 255 * config['edge_darkratio']
        edge_mask = np.repeat(edge_mask[:, :, np.newaxis], 3, axis=2)
        result = (result * (1 - edge_mask)).astype(np.uint8)
    
    return result

def apply_raindrop_effect(image_path, drop_config=None):
    """
    Apply raindrop effect to an image.
    
    Args:
        image_path: Path to the image file
        drop_config: Configuration for raindrop effect (optional)
    
    Returns:
        Image with raindrop effect applied
    """
    if drop_config is None:
        drop_config = {
            'maxR': 50,
            'minR': 30,
            'maxDrops': 30,
            'minDrops': 30,
            'edge_darkratio': 0.3,
        }
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Add raindrops
    result = add_raindrop(image, drop_config)
    
    # Convert to PIL Image for saving
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    return result_pil

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
    raindrop_config = {
        'maxR': 50,
        'minR': 30,
        'maxDrops': 30,
        'minDrops': 30,
        'edge_darkratio': 0.3,
    }
    
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