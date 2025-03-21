"""
A Python implementation of Perlin noise for the fog effect.
This is a simplified version that doesn't require C++ build tools.
"""
import numpy as np

def generate_perlin_noise(width, height, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate a 2D Perlin noise array
    
    Args:
        width: Width of the output array
        height: Height of the output array
        scale: Scale of the noise (smaller values = more zoomed out)
        octaves: Number of layers of noise
        persistence: How much each octave contributes to the overall shape
        lacunarity: How much detail is added in each octave
        seed: Random seed for reproducibility
        
    Returns:
        2D numpy array of Perlin noise values in range [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.zeros((height, width))
    
    # Generate Perlin noise by summing octaves
    max_amplitude = 0
    amplitude = 1.0
    frequency = 1.0
    
    # Generate a grid of random gradient vectors
    def generate_gradients(shape):
        angles = 2 * np.pi * np.random.random(shape)
        return np.dstack((np.cos(angles), np.sin(angles)))
    
    for i in range(octaves):
        # Calculate the number of grid points
        grid_width = int(width // (scale / frequency)) + 2
        grid_height = int(height // (scale / frequency)) + 2
        
        # Generate random gradient vectors at each grid point
        gradients = generate_gradients((grid_height, grid_width))
        
        # For each pixel, calculate the contribution from this octave
        for y in range(height):
            for x in range(width):
                # Calculate the position in the grid
                x_grid = x / (scale / frequency)
                y_grid = y / (scale / frequency)
                
                # Get the grid cell coordinates
                x0 = int(x_grid)
                y0 = int(y_grid)
                x1 = x0 + 1
                y1 = y0 + 1
                
                # Get the fractional part
                dx = x_grid - x0
                dy = y_grid - y0
                
                # Interpolation weights
                sx = dx * dx * (3 - 2 * dx)
                sy = dy * dy * (3 - 2 * dy)
                
                # Calculate dot products between gradients and distance vectors
                n0 = np.dot(gradients[y0, x0], [dx, dy])
                n1 = np.dot(gradients[y0, x1], [dx - 1, dy])
                ix0 = n0 + sx * (n1 - n0)
                
                n0 = np.dot(gradients[y1, x0], [dx, dy - 1])
                n1 = np.dot(gradients[y1, x1], [dx - 1, dy - 1])
                ix1 = n0 + sx * (n1 - n0)
                
                value = ix0 + sy * (ix1 - ix0)
                
                # Scale from [-1, 1] to [0, 1]
                value = (value + 1) / 2
                
                # Add to noise
                noise[y, x] += value * amplitude
        
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalize to [0, 1]
    return noise / max_amplitude

def generate_noise_3d(width, height, depth, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate a 2D noise image based on x, y coordinates and depth value.
    This is a simplification of 3D noise where we use the depth as a seed offset.
    
    Args:
        width: Width of the output array
        height: Height of the output array
        depth: Depth value to use for the third dimension
        scale: Scale of the noise
        octaves: Number of layers of noise
        persistence: How much each octave contributes to the overall shape
        lacunarity: How much detail is added in each octave
        seed: Random seed for reproducibility
        
    Returns:
        2D numpy array of noise values in range [0, 1]
    """
    if seed is None:
        seed = np.random.randint(0, 1000)
    
    # Use depth to modify the seed
    seed_offset = int(depth * 1000) % 1000
    modified_seed = (seed + seed_offset) % 2**32
    
    return generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, modified_seed) 