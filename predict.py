import torch
from torchvision import transforms
from PIL import Image
import os
import random
from src.model import BrainTumorModel

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("saved_models", "best_model_finetuned.pth")
DATA_DIR = os.path.join("data", "raw", "MRI_images", "Testing")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def load_model():
    """Loads the trained model from disk."""
    model = BrainTumorModel(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict_single_image(image_path, model):
    """Performs inference on a single image file."""
    # Standard Clinical Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return "ERROR", 0.0

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        
    return CLASSES[predicted_class.item()], confidence.item()

def run_batch_diagnosis(model):
    """Runs a random sample test across all categories."""
    print(f"\n{'='*20} SYSTEM DIAGNOSTIC (BATCH MODE) {'='*20}")
    print(f"{'FILENAME':<25} | {'TRUE LABEL':<12} | {'PREDICTION':<12} | {'CONF':<6} | {'STATUS'}")
    print("-" * 85)
    
    total_correct = 0
    total_samples = 0
    
    # Iterate over every tumor category
    for true_label in CLASSES:
        folder_path = os.path.join(DATA_DIR, true_label)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            continue
            
        # Get random sample of 3 images
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) < 3: sample = files
        else: sample = random.sample(files, 3)
        
        for img_name in sample:
            img_path = os.path.join(folder_path, img_name)
            pred_label, conf = predict_single_image(img_path, model)
            
            is_correct = (pred_label == true_label)
            status = "✅" if is_correct else "❌"
            if is_correct: total_correct += 1
            total_samples += 1
            
            print(f"{img_name[:23]:<25} | {true_label:<12} | {pred_label:<12} | {conf*100:.0f}%   | {status}")

    print("-" * 85)
    print(f"DIAGNOSTIC RESULT: {total_correct}/{total_samples} Correct")
    print(f"{'='*66}\n")

if __name__ == "__main__":
    # 1. Load the Brain
    if os.path.exists(MODEL_PATH):
        print(f"Loading clinical model from: {MODEL_PATH}")
        model = load_model()
        
        # 2. Run the Batch Test (Default behavior)
        # This will test 3 random images from EACH folder (Glioma, Meningioma, etc.)
        run_batch_diagnosis(model)
        
    else:
        print(f"Error: Model not found at {MODEL_PATH}. Please run train.py first.")