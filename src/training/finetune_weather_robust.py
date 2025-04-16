import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path

from PIL import Image
from torchvision import models, transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Import utilities from evaluation script
sys.path.append(str(Path(__file__).resolve().parents[1]))
from evaluation.evaluatePretrained import get_default_device, to_device, DeviceDataLoader, clear_gpu_memory
from evaluation.evaluatePretrained import accuracy, validation_step, validation_epoch_end, evaluate
from evaluation.evaluatePretrained import obtain_performance_metrics

class CombinedPlantDataset(ImageFolder):
    """Dataset that includes class names for better reporting"""
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.samples_source = ['clean'] * len(self.samples)  # Default source is 'clean'
    
    def mark_samples_source(self, source_name):
        """Mark all samples in this dataset with a source identifier"""
        self.samples_source = [source_name] * len(self.samples)
        return self
    
    def __getitem__(self, index):
        """Override to return the source along with the sample"""
        img, label = super().__getitem__(index)
        return img, label

def train_step(model, batch, optimizer):
    """Perform one training step"""
    model.train()
    images, labels = batch
    
    # Forward pass
    out = model(images)
    loss = F.cross_entropy(out, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Return loss and accuracy
    acc = accuracy(out, labels)
    return {'train_loss': loss.detach(), 'train_accuracy': acc}

def train_epoch(model, train_loader, optimizer):
    """Train for one epoch"""
    batch_losses = []
    batch_accs = []
    
    # Process each batch
    for batch in tqdm(train_loader, desc="Training"):
        results = train_step(model, batch, optimizer)
        batch_losses.append(results['train_loss'])
        batch_accs.append(results['train_accuracy'])
    
    # Calculate epoch metrics
    epoch_loss = torch.stack(batch_losses).mean()
    epoch_acc = torch.stack(batch_accs).mean()
    
    return {'train_loss': epoch_loss, 'train_accuracy': epoch_acc}

def finetune_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=5, 
    learning_rate=0.0001, 
    weight_decay=1e-4
):
    """Finetune a pretrained model with weather-augmented images"""
    # Define optimizer - use a small learning rate for finetuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize best accuracy
    best_acc = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_result = train_epoch(model, train_loader, optimizer)
        
        # Validate
        val_result = evaluate(model, val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_result['train_loss']:.4f}, Train Acc: {train_result['train_accuracy']:.4f}")
        print(f"Val Loss: {val_result['val_loss']:.4f}, Val Acc: {val_result['val_accuracy']:.4f}")
        print("=" * 50)
        
        # Save best model
        if val_result['val_accuracy'] > best_acc:
            best_acc = val_result['val_accuracy']
            best_model_state = model.state_dict()
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    """Main function to finetune and evaluate the model"""
    # Set device
    device = get_default_device()
    clear_gpu_memory()
    
    print(f"Using device: {device}")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load the pretrained model
    model_path = "src/models/pre-trained/resnet50-pre-trained-plants-afarturc.pth"
    
    pretrained_model = models.resnet50(pretrained=False)
    num_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_features, 38)  # 38 classes
    
    # Load the pretrained weights
    loaded_state = torch.load(model_path, map_location=device)
    pretrained_model.load_state_dict(loaded_state)
    
    # Move model to device
    pretrained_model.to(device)
    
    # Define data transformations
    # For training - include augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    
    # For validation/testing - only resize and normalize
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Setup paths for datasets
    base_dir = Path("data/plant_village_limited_split")
    
    # Clean datasets
    clean_train_dir = base_dir / "processed/train"
    clean_val_dir = base_dir / "processed/val"
    clean_test_dir = base_dir / "processed/test"
    
    # Weather-augmented datasets
    fog_train_dir = base_dir / "augmented/weather/fog/train"
    raindrop_train_dir = base_dir / "augmented/weather/raindrop/train"
    
    fog_test_dir = base_dir / "augmented/weather/fog/test"
    raindrop_test_dir = base_dir / "augmented/weather/raindrop/test"
    
    # Load datasets
    print("Loading datasets...")
    
    # Training sets
    clean_train_dataset = CombinedPlantDataset(root=clean_train_dir, transform=train_transform).mark_samples_source('clean')
    fog_train_dataset = CombinedPlantDataset(root=fog_train_dir, transform=train_transform).mark_samples_source('fog')
    raindrop_train_dataset = CombinedPlantDataset(root=raindrop_train_dir, transform=train_transform).mark_samples_source('raindrop')
    
    # Validation set (we'll use the clean validation set)
    val_dataset = CombinedPlantDataset(root=clean_val_dir, transform=test_transform)
    
    # Test sets
    clean_test_dataset = CombinedPlantDataset(root=clean_test_dir, transform=test_transform)
    fog_test_dataset = CombinedPlantDataset(root=fog_test_dir, transform=test_transform)
    raindrop_test_dataset = CombinedPlantDataset(root=raindrop_test_dir, transform=test_transform)
    
    # Combine datasets for training
    combined_train_dataset = ConcatDataset([clean_train_dataset, fog_train_dataset, raindrop_train_dataset])
    
    # Create data loaders
    batch_size = 32
    
    # Training loader with combined datasets
    train_loader = DataLoader(
        combined_train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Convert to device data loaders
    train_loader = DeviceDataLoader(train_loader, device)
    
    # Validation loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DeviceDataLoader(val_loader, device)
    
    # Test loaders
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    clean_test_loader = DeviceDataLoader(clean_test_loader, device)
    
    fog_test_loader = DataLoader(fog_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    fog_test_loader = DeviceDataLoader(fog_test_loader, device)
    
    raindrop_test_loader = DataLoader(raindrop_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    raindrop_test_loader = DeviceDataLoader(raindrop_test_loader, device)
    
    # Get class names
    class_names = val_dataset.classes
    
    # Print dataset statistics
    print(f"Clean training samples: {len(clean_train_dataset)}")
    print(f"Fog training samples: {len(fog_train_dataset)}")
    print(f"Raindrop training samples: {len(raindrop_train_dataset)}")
    print(f"Total training samples: {len(combined_train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("=" * 50)
    
    # Finetune the model
    print("Starting finetuning...")
    finetune_epochs = 10  # Adjust as needed
    finetuned_model = finetune_model(
        model=pretrained_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=finetune_epochs,
        learning_rate=0.0001,  # Small learning rate for finetuning
        weight_decay=1e-4
    )
    
    # Save the finetuned model
    output_dir = Path("src/models/finetuned")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_save_path = output_dir / "resnet50_weather_robust.pth"
    torch.save(finetuned_model.state_dict(), model_save_path)
    print(f"Saved finetuned model to {model_save_path}")
    print("=" * 50)
    
    # Evaluate on test sets
    print("Evaluating model on test sets...")
    
    # 1. Clean Test Set
    print("\n=== Clean Test Set Results ===")
    clean_results = evaluate(finetuned_model, clean_test_loader)
    print(f"Clean Test Accuracy: {clean_results['val_accuracy'].item():.4f}")
    print(f"Clean Test Loss: {clean_results['val_loss'].item():.4f}")
    obtain_performance_metrics(clean_results["ground_truths"], clean_results["predictions"], class_names)
    
    # 2. Fog Test Set
    print("\n=== Foggy Test Set Results ===")
    fog_results = evaluate(finetuned_model, fog_test_loader)
    print(f"Fog Test Accuracy: {fog_results['val_accuracy'].item():.4f}")
    print(f"Fog Test Loss: {fog_results['val_loss'].item():.4f}")
    obtain_performance_metrics(fog_results["ground_truths"], fog_results["predictions"], class_names)
    
    # 3. Raindrop Test Set
    print("\n=== Rainy Test Set Results ===")
    raindrop_results = evaluate(finetuned_model, raindrop_test_loader)
    print(f"Raindrop Test Accuracy: {raindrop_results['val_accuracy'].item():.4f}")
    print(f"Raindrop Test Loss: {raindrop_results['val_loss'].item():.4f}")
    obtain_performance_metrics(raindrop_results["ground_truths"], raindrop_results["predictions"], class_names)
    
    print("\nDone! Weather-robust model finetuning and evaluation complete.")

if __name__ == "__main__":
    main()