import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.data_loader import get_data_loaders
from src.model import BrainTumorModel

DATA_PATH = os.path.join("data", "raw", "MRI_images") 
EPOCHS = 10           
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc="Fine-Tuning")
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def main():
    print(f"Fine-Tuning on Device: {DEVICE}")
    
    # 1. Load Data
    train_loader, val_loader, _, class_names = get_data_loaders(DATA_PATH, BATCH_SIZE)
    
    # 2. Load the Previous Best Model
    print("Loading previous best model...")
    model = BrainTumorModel(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load("saved_models/best_model.pth"))
    
    # 3. UNFREEZE the weights (The Magic Step)
    print("Unfreezing feature layers for fine-tuning...")
    for param in model.backbone.features.parameters():
        param.requires_grad = True
        
    # 4. Setup Loss and Optimizer with LOWER Learning Rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    best_acc = 0.0
    
    # Check baseline before we start
    print("Checking baseline accuracy...")
    val_loss, val_acc = validate(model, val_loader, criterion)
    print(f"Starting Baseline: {val_acc:.2f}%")
    best_acc = val_acc

    for epoch in range(EPOCHS):
        print(f"\nFine-Tune Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("saved_models", "best_model_finetuned.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Improved Model Saved! ({val_acc:.2f}%)")

if __name__ == "__main__":
    main()