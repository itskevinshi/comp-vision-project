import os
import cv2
import numpy as np
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

# Import our custom Perlin noise implementation
from src.weather_filters.perlin import generate_perlin_noise, generate_noise_3d

# Add the FoHIS module to the Python path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels to reach project root
sys.path.append(str(project_root / "models" / "FoHIS-master" / "FoHIS"))

# Import FoHIS modules if available, otherwise use our constants
try:
    import tool_kit as tk
    from parameter import const
except ImportError:
    # Define constants if the FoHIS module is not available
    class Const:
        def __init__(self):
            # Default values based on FoHIS parameters
            self.VISIBILITY_RANGE_MOLECULE = 12  # m
            self.VISIBILITY_RANGE_AEROSOL = 450  # m
            self.ECM = 3.912 / self.VISIBILITY_RANGE_MOLECULE
            self.ECA = 3.912 / self.VISIBILITY_RANGE_AEROSOL
            self.FT = 70  # FOG_TOP m
            self.HT = 34  # HAZE_TOP m
            self.CAMERA_ALTITUDE = 1.8  # m
            self.HORIZONTAL_ANGLE = 0  # degrees
            self.CAMERA_VERTICAL_FOV = 64  # degrees
    
    const = Const()
    tk = None

def custom_noise(image, depth):
    """
    Generate Perlin noise for fog effect using our custom implementation.
    This replaces the tk.noise function from FoHIS.
    
    Args:
        image: RGB image as numpy array
        depth: Depth map as numpy array
        
    Returns:
        Perlin noise as numpy array
    """
    height, width = image.shape[:2]
    
    # Generate three layers of noise at different scales
    p1 = generate_perlin_noise(width, height, scale=130.0, octaves=1, persistence=0.5, lacunarity=2.0) * 255
    p2 = generate_perlin_noise(width, height, scale=60.0, octaves=1, persistence=0.5, lacunarity=2.0) * 255
    p3 = generate_perlin_noise(width, height, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0) * 255
    
    # Combine the noise layers with weights
    perlin = (p1 + p2/2 + p3/4) / 3
    
    return perlin

def custom_elevation_distance(img, depth, vertical_fov, horizontal_angle, camera_altitude):
    """
    Simplified version of elevation_and_distance_estimation from FoHIS toolkit.
    
    Args:
        img: RGB image
        depth: Depth map
        vertical_fov: Camera vertical field of view
        horizontal_angle: Camera horizontal angle
        camera_altitude: Camera altitude
        
    Returns:
        Altitude, distance, and angle maps
    """
    height, width = img.shape[:2] if isinstance(img, np.ndarray) else img
    altitude = np.ones((height, width)) * camera_altitude
    distance = depth.copy()
    angle = np.zeros((height, width))
    
    return altitude, distance, angle

def apply_fog_effect(image, depth=None):
    """
    Apply fog effect to an image using a simplified version of the FoHIS implementation.
    
    Args:
        image: RGB image as a numpy array
        depth: Depth map as a numpy array (optional). If not provided, a simple depth
               estimate will be created based on image dimensions.
    
    Returns:
        Foggy image as a numpy array
    """
    height, width = image.shape[:2]
    
    # If no depth map is provided, create a simple one
    # For leaf images, we'll assume a gradual depth from center to edges
    if depth is None:
        depth = np.zeros((height, width), dtype=np.float64)
        center_y, center_x = height // 2, width // 2
        
        for y in range(height):
            for x in range(width):
                # Calculate distance from center and normalize to create depth
                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                max_dist = np.sqrt(center_y**2 + center_x**2)
                depth[y, x] = 30 * (1 - dist / max_dist) + 1  # Scale to [1, 31]
    
    depth = depth.astype(np.float64)
    depth[depth == 0] = 1  # The depth_min shouldn't be 0
    
    # Determine which elevation and distance function to use
    if tk is not None:
        try:
            # Create a temporary file for the image since toolkit expects a file path
            temp_img_path = str(project_root / "temp_img.jpg")
            cv2.imwrite(temp_img_path, image)
            
            # Estimate elevation and distance using the original function
            elevation, distance, angle = tk.elevation_and_distance_estimation(
                temp_img_path, depth,
                const.CAMERA_VERTICAL_FOV,
                const.HORIZONTAL_ANGLE,
                const.CAMERA_ALTITUDE
            )
            
            # Remove temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        except Exception as e:
            print(f"Warning: Using simplified elevation calculation due to error: {e}")
            elevation, distance, angle = custom_elevation_distance(
                image, depth,
                const.CAMERA_VERTICAL_FOV,
                const.HORIZONTAL_ANGLE,
                const.CAMERA_ALTITUDE
            )
    else:
        elevation, distance, angle = custom_elevation_distance(
            image, depth,
            const.CAMERA_VERTICAL_FOV,
            const.HORIZONTAL_ANGLE,
            const.CAMERA_ALTITUDE
        )
    
    # Initialize arrays
    I = np.empty_like(image)
    result = np.empty_like(image)
    
    if const.FT != 0:
        try:
            # Use FoHIS noise function if available, otherwise use our custom one
            if tk is not None and hasattr(tk, 'noise'):
                perlin = tk.noise(image, depth)
            else:
                perlin = custom_noise(image, depth)
                
            ECA = const.ECA
            c = (1 - elevation / (const.FT + 0.00001))
            c[c < 0] = 0
            
            if const.FT > const.HT:
                ECM = (const.ECM * c + (1 - c) * const.ECA) * (perlin / 255)
            else:
                ECM = (const.ECA * c + (1 - c) * const.ECM) * (perlin / 255)
        except Exception as e:
            print(f"Warning: Using simplified noise calculation due to error: {e}")
            ECA = const.ECA
            ECM = const.ECM
    else:
        ECA = const.ECA
        ECM = const.ECM
    
    # Calculate fog effect based on the FoHIS implementation
    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)
    
    # Using just the fog implementation for simplicity
    # Fog-only effect (simplified from the original implementation)
    idx2 = elevation <= const.FT
    distance_through_fog[idx2] = distance[idx2]
    
    I[:, :, 0] = image[:, :, 0] * np.exp(-ECM * distance_through_fog)
    I[:, :, 1] = image[:, :, 1] * np.exp(-ECM * distance_through_fog)
    I[:, :, 2] = image[:, :, 2] * np.exp(-ECM * distance_through_fog)
    O = 1 - np.exp(-ECM * distance_through_fog)
    
    # Color of the fog (grayish white)
    Ial = np.empty_like(image)
    Ial[:, :, 0] = 225  # B
    Ial[:, :, 1] = 225  # G
    Ial[:, :, 2] = 201  # R
    
    result[:, :, 0] = I[:, :, 0] + O * Ial[:, :, 0]
    result[:, :, 1] = I[:, :, 1] + O * Ial[:, :, 1]
    result[:, :, 2] = I[:, :, 2] + O * Ial[:, :, 2]
    
    return result

def process_dataset(input_dir, output_dir, fog_intensity='medium'):
    """
    Process all images in the input directory and save foggy versions to output directory.
    
    Args:
        input_dir: Directory containing clean images
        output_dir: Directory to save foggy images
        fog_intensity: 'light', 'medium', or 'heavy'
    """
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
        
        # Process all images in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in tqdm(image_files):
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

def main(fog_intensity='medium'):
    """
    Main function to apply fog augmentation to the dataset.
    
    Args:
        fog_intensity: 'light', 'medium', or 'heavy'
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
        process_dataset(str(input_dir), str(output_dir), fog_intensity=fog_intensity)
    
    print("\nFog augmentation completed!")

if __name__ == "__main__":
    # If script is run directly, use medium intensity by default
    main('medium') 