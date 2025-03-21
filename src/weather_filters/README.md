# Weather Filters for Image Augmentation

This directory contains implementations of weather effects that can be applied to plant disease images to simulate challenging weather conditions.

## Available Filters

### Fog Effect

The fog effect is based on the FoHIS (Foggy and Hazy Image Simulation) implementation from:

> **Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity**  
> Ning Zhang, Lin Zhang, and Zaixi Cheng  
> International Conference on Neural Information Processing, 2017

The implementation can be found in the `models/FoHIS-master` directory. Our adaptation:

1. Simplifies the original implementation for our use case
2. Provides different fog intensity options (light, medium, heavy)
3. Generates a synthetic depth map for leaf images

#### Alternative Implementation

We've also included a pure Python implementation of the fog effect that doesn't require the FoHIS module or C++ build tools. This implementation:

1. Uses a custom Perlin noise generator (`perlin.py`) instead of relying on the external `noise` package
2. Automatically falls back to this implementation if FoHIS is not available
3. Produces similar results but may be slower for large images

## Usage

To apply the fog effect to the processed dataset, run:

```bash
python src/main.py fog --intensity [light|medium|heavy]
```

This will process all images in the `data/processed/{train,val,test}` directories and save the foggy versions to `data/augmented/weather/fog/{train,val,test}`.

## Implementation Details

### Fog Effect

The fog effect is implemented in `apply_fog.py`. Key components:

1. **Depth Map Generation**: For leaf images where we don't have actual depth information, we generate a synthetic depth map assuming the center of the leaf is closer to the camera.

2. **Fog Parameters**: 
   - Light fog: Visibility range = 24m, Fog top = 40m
   - Medium fog: Visibility range = 12m, Fog top = 70m
   - Heavy fog: Visibility range = 6m, Fog top = 100m

3. **Process**: The effect applies atmospheric scattering principles to simulate how light is attenuated through fog, creating a realistic fog effect on the images.

4. **Perlin Noise**: The custom implementation in `perlin.py` generates noise patterns that simulate the non-uniform nature of fog, creating more realistic results.

## Future Work

- [ ] Rain effect implementation
- [ ] Snow effect implementation
- [ ] Night/low-light conditions 