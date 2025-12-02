import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from src.model import BrainTumorModel

DATA_PATH = os.path.join("data", "raw", "MRI_images")
MODEL_PATH = os.path.join("saved_models", "best_model_finetuned.pth") 
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_test_loader(data_dir):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_dir = os.path.join(data_dir, 'Testing')
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, test_dataset.classes

def evaluate_model():
    print(f"Evaluating on Device: {DEVICE}")
    
    # 1. Load Data
    test_loader, class_names = get_test_loader(DATA_PATH)
    
    # 2. Load Model
    print("Loading Model...")
    model = BrainTumorModel(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Get Predictions
    y_true = []
    y_pred = []
    
    print("Running Inference on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())
            
    # 4. Generate Report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 5. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Brain Tumor Detection')
    plt.show()

if __name__ == "__main__":
    evaluate_model()