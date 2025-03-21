import os
import shutil
from pathlib import Path

# Paths to train/val/test directories
input_dirs = ['plant_village_limited/train', 'plant_village_limited/val', 'plant_village_limited/test']
recombined_dir = Path('recombined_plant_village_limited')

# Clean and prepare the recombined output folder
if recombined_dir.exists():
    shutil.rmtree(recombined_dir)
recombined_dir.mkdir(parents=True, exist_ok=True)

# Go through each split
for input_dir in input_dirs:
    for category in os.listdir(input_dir):
        category_path = Path(input_dir) / category
        if category_path.is_dir():
            target_category_path = recombined_dir / category
            target_category_path.mkdir(parents=True, exist_ok=True)

            for image in category_path.glob('*'):
                # Prevent name collisions by renaming duplicates if necessary
                dst_path = target_category_path / image.name
                counter = 1
                while dst_path.exists():
                    dst_path = target_category_path / f"{image.stem}_{counter}{image.suffix}"
                    counter += 1
                shutil.copy2(image, dst_path)

print("✅ Images recombined into 'recombined_dataset' by category.")
