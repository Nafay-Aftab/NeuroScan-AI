import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from src.data_loader import get_data_loaders

DATA_PATH = os.path.join("data", "raw", "MRI_images") 

def denormalize(tensor):
    """Reverses the ImageNet normalization so we can view the image normally."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    tensor = tensor.numpy().transpose((1, 2, 0)) # Convert (C, H, W) to (H, W, C)
    tensor = std * tensor + mean                 # Un-normalize
    tensor = np.clip(tensor, 0, 1)               # Clip values
    return tensor

def visualize_batch():
    print(f"Looking for data at: {os.path.abspath(DATA_PATH)}")
    
    try:
        train_loader, _, _, class_names = get_data_loaders(DATA_PATH, batch_size=4)
    except FileNotFoundError:
        print("ERROR: Could not find the dataset. Please check if you moved 'MRI_images' into 'data/raw/' correctly.")
        return

    print(f"Classes Detected: {class_names}")
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        ax = axes[i]
        img = denormalize(images[i])
        label = class_names[labels[i]]
        
        ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')
    
    plt.show()

if __name__ == "__main__":
    visualize_batch()
