#!/usr/bin/env python3
"""
Run finetuning of a pretrained model with weather-augmented images for improved robustness.
"""
import os
import sys
import argparse
import torch
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from training.finetune_weather_robust import main as finetune_main

if __name__ == "__main__":
    # Check for CUDA availability only when script is run directly
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    parser = argparse.ArgumentParser(description='Finetune a pretrained model with weather-augmented images')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for finetuning (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate for finetuning (default: 0.0001)')
    
    args = parser.parse_args()
    
    # Make sure the required directories exist
    os.makedirs("src/models/finetuned", exist_ok=True)
    
    # Call the main finetuning function
    print("Starting weather-robust model finetuning...")
    finetune_main()