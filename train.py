import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bars
import os

# Import our custom modules
from src.data_loader import get_data_loaders
from src.model import BrainTumorModel

# CONSTANTS
DATA_PATH = os.path.join("data", "raw", "MRI_images") 
EPOCHS = 3           # How many times to look at the entire dataset
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, loader, criterion, optimizer):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm creates a nice progress bar in the terminal
    loop = tqdm(loader, desc="Training")
    
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # 1. Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward Pass
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Calculate new gradients
        optimizer.step()      # Update weights
        
        # 3. Metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item()) # Update progress bar
        
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion):
    model.eval() # Set model to evaluation mode (freezes Dropout, BatchNorm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # Don't calculate gradients (saves RAM/Speed)
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
    print(f"Using Device: {DEVICE}")
    
    # 1. Setup Data
    print("Loading Data...")
    train_loader, val_loader, _, class_names = get_data_loaders(DATA_PATH, BATCH_SIZE)
    
    # 2. Setup Model
    print("Initializing Model...")
    model = BrainTumorModel(num_classes=len(class_names)).to(DEVICE)
    
    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    best_acc = 0.0
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("saved_models", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Model Saved! ({val_acc:.2f}%)")

if __name__ == "__main__":
    main()