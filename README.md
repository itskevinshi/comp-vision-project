# Improving Computer Vision Methods for Plant Pathology Detection

A research project focused on developing improved computer vision methods for plant disease detection, with a particular focus on robustness to adverse weather conditions.

## Project Overview

This project aims to address the challenge of detecting plant diseases under various environmental conditions, including adverse weather like rain and fog. Farmers often need to diagnose plant diseases in less-than-ideal conditions, and our goal is to develop models that can perform well regardless of weather conditions.

## Directory Structure

- `data/`
  - `raw/` - Original PlantVillage dataset
  - `processed/` - Preprocessed images split into train, validation, and test sets
  - `augmented/` - Augmented datasets
    - `standard/` - Standard augmentations (geometric transformations, color adjustments)
    - `weather/` - Weather-based augmentations (fog, rain)
    - `combined/` - Combined standard and weather augmentations

- `models/` - Model implementations and external repositories
  - `FoHIS-master/` - Fog and Haze Image Simulation implementation

- `src/` - Source code
  - `data/` - Data processing scripts
  - `models/` - Model implementation
  - `training/` - Training scripts
  - `evaluation/` - Evaluation scripts
  - `weather_filters/` - Weather augmentation implementations

- `notebooks/` - Jupyter notebooks for analysis and visualization

- `configs/` - Configuration files

- `outputs/` - Model outputs and results

## Usage

You can use the main entry point to run different parts of the project:

```bash
python src/main.py [command] [options]
```

Available commands:

- `fog`: Apply fog augmentation to the dataset
  ```bash
  python src/main.py fog --intensity [light|medium|heavy]
  ```

- `check-deps`: Check if dependencies are installed
  ```bash
  python src/main.py check-deps
  ```

## Weather Augmentation

### Fog Effect

We've implemented a fog augmentation based on the FoHIS (Foggy and Hazy Image Simulation) method. This generates realistic fog effects on plant images to simulate challenging weather conditions.

#### Implementation Notes

- The implementation primarily uses the FoHIS method when available
- A pure Python fallback implementation is provided that doesn't require C++ build tools
- The fallback uses a custom Perlin noise generator for fog texture

To apply fog augmentation to the dataset:

```bash
python src/main.py fog --intensity medium
```

The augmented images will be saved to `data/augmented/weather/fog/`.

## Setup and Dependencies

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Check dependencies:

```bash
python src/main.py check-deps
```

Note: The C++ build dependency for the `noise` package has been removed. We now include a custom Python implementation that doesn't require C++ compilation.

## Team

- Bryson M (han3wf)
- Alex Talreja (vta3nc)
- Kevin Shi (nwk6tq) 