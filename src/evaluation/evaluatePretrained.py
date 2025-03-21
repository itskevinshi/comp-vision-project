#########################################################
# 1. Imports and Basic Setup
#########################################################
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

#########################################################
# 2. Device Utilities
#########################################################
def get_default_device():
    """Pick GPU if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(data, device):
    """Move tensor(s) to chosen device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a DataLoader to move data to a device."""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)
        
    def __len__(self):
        """Number of batches."""
        return len(self.dl)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

#########################################################
# 3. Accuracy and Validation Helpers
#########################################################
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def validation_step(model, batch):
    predictions = []
    ground_truths = [] 
    
    images, labels = batch
    out = model(images)                    # Generate prediction
    loss = F.cross_entropy(out, labels)    # Calculate loss
    acc = accuracy(out, labels)            # Calculate accuracy

    _, predicted = torch.max(out, 1)
    predictions.extend(predicted.cpu().numpy())
    ground_truths.extend(labels.cpu().numpy())

    return {
        "val_loss": loss.detach(),
        "val_accuracy": acc,
        "predictions": predictions,
        "ground_truths": ground_truths
    }

@torch.no_grad()
def validation_epoch_end(outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    batch_accuracy = [x["val_accuracy"] for x in outputs]

    epoch_loss = torch.stack(batch_losses).mean()
    epoch_accuracy = torch.stack(batch_accuracy).mean()

    predictions = []
    ground_truths = []

    for x in outputs:
        predictions.extend(x["predictions"])
        ground_truths.extend(x["ground_truths"])

    return {
        "val_loss": epoch_loss, 
        "val_accuracy": epoch_accuracy,
        "predictions": predictions,
        "ground_truths": ground_truths
    }

@torch.no_grad()
def evaluate(model, val_loader):
    """Evaluate the model on the entire loader, returning performance stats."""
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(outputs)

def obtain_performance_metrics(ground_truths, predictions, class_names):
    """Show classification report, balanced accuracy, and confusion matrix."""
    print("Classification Report:\n")
    print(classification_report(ground_truths, predictions, target_names=class_names))
    print("Balanced accuracy score:", balanced_accuracy_score(ground_truths, predictions))
    
    matrix = confusion_matrix(ground_truths, predictions)
    plt.figure(figsize=(10,10))
    sns.heatmap(matrix, cmap="PuBu", annot=True, linewidths=0.5, fmt='d', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.show()

#########################################################
# 4. MAIN: Load Model & Evaluate on Augmented Data
#########################################################
if __name__ == "__main__":
    device = get_default_device()
    clear_gpu_memory()

    # -----------------------------------------------------
    # 4.1 Load the Pretrained Model
    # -----------------------------------------------------
    # Path to your .pth file (make sure it exists here)
    model_path = "src/models/pre-trained/resnet50-pre-trained-plants-afarturc.pth"
    

    # Create a resnet50 and manually set the final fc to 38 outputs
    pretrained_model = models.resnet50(pretrained=False)
    num_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_features, 38)  # 38 classes

    # Load the author's checkpoint
    loaded_state = torch.load(model_path, map_location=device)
    pretrained_model.load_state_dict(loaded_state)

    pretrained_model.to(device)
    pretrained_model.eval()
    # -----------------------------------------------------
    # 4.2 Evaluate on RAIN test set
    # -----------------------------------------------------
    # Make sure your subfolders: data/augmented/weather/rain/test/<class_subfolders>
    # match the same classes the model was trained on
    rain_test_dir = "plant_village_limited/test"

    # Transforms: must match the author's input pipeline (e.g. 256x256)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    rain_test_dataset = ImageFolder(rain_test_dir, transform=test_transform)
    # We'll get the class names from this dataset:
    class_names = rain_test_dataset.classes

    rain_test_loader = DataLoader(rain_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    rain_test_loader = DeviceDataLoader(rain_test_loader, device)

    # Evaluate
    rain_results = evaluate(pretrained_model, rain_test_loader)
    print("=== Rainy Test Set Results ===")
    print(f"Rain Test Accuracy: {rain_results['val_accuracy'].item():.4f}")
    print(f"Rain Test Loss: {rain_results['val_loss'].item():.4f}")

    # Metrics: confusion matrix, classification report, etc.
    obtain_performance_metrics(rain_results["ground_truths"], rain_results["predictions"], class_names)

    # -----------------------------------------------------
    # 4.3 Evaluate on FOG test set
    # -----------------------------------------------------
    fog_test_dir = "data/augmented/weather/fog/test"
    fog_test_dataset = ImageFolder(fog_test_dir, transform=test_transform)
    # class_names should be the same, but we can check:
    # if fog_test_dataset.classes != class_names, there's a mismatch
    fog_test_loader = DataLoader(fog_test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    fog_test_loader = DeviceDataLoader(fog_test_loader, device)

    fog_results = evaluate(pretrained_model, fog_test_loader)
    print("=== Foggy Test Set Results ===")
    print(f"Fog Test Accuracy: {fog_results['val_accuracy'].item():.4f}")
    print(f"Fog Test Loss: {fog_results['val_loss'].item():.4f}")

    obtain_performance_metrics(fog_results["ground_truths"], fog_results["predictions"], class_names)

    print("\nDone! Model evaluation on augmented images complete.")
