import os
import shutil
import random
from pathlib import Path

# Paths to original folders
original_train_dir = Path('../PlantVillage/train')
original_val_dir = Path('../PlantVillage/val')

# Output folders
output_base = Path('plant_village_limited_split')
output_train = output_base / 'train'
output_val = output_base / 'val'
output_test = output_base / 'test'

# Temporary folder for merged categories
merged_dir = Path('merged_temp')

# Ensure clean state
if merged_dir.exists():
    shutil.rmtree(merged_dir)
if output_base.exists():
    shutil.rmtree(output_base)

# Helper: get only folder names
def get_category_folders(path):
    return [f for f in os.listdir(path) if (path / f).is_dir()]

# Step 1: Get all unique category folders
category_folders = sorted(set(
    get_category_folders(original_train_dir) +
    get_category_folders(original_val_dir)
))

# Step 2: Merge images of each category
os.makedirs(merged_dir, exist_ok=True)

for category in category_folders:
    merged_category_path = merged_dir / category
    os.makedirs(merged_category_path, exist_ok=True)

    combined_images = []

    for source_dir in [original_train_dir, original_val_dir]:
        category_path = source_dir / category
        if category_path.is_dir():
            images = list(category_path.glob('*'))
            combined_images.extend(images)

    # Shuffle and limit to 100 images max
    random.shuffle(combined_images)
    combined_images = combined_images[:100]

    for img in combined_images:
        dst = merged_category_path / img.name
        shutil.copy2(img, dst)

# Step 3: Split into train/val/test (70/15/15)
for category in os.listdir(merged_dir):
    all_images = list((merged_dir / category).glob('*'))
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val  # Account for rounding

    splits = {
        output_train / category: all_images[:n_train],
        output_val / category: all_images[n_train:n_train + n_val],
        output_test / category: all_images[n_train + n_val:]
    }

    for split_dir, files in splits.items():
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            shutil.copy2(f, split_dir / f.name)

# Cleanup
shutil.rmtree(merged_dir)

print("Dataset merged, capped to 100 per class, and split into train/val/test.")
