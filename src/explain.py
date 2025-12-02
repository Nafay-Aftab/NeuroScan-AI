import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from src.model import BrainTumorModel

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join("saved_models", "best_model_finetuned.pth")

# --- GRAD-CAM ENGINE ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook the gradients
        target_layer.register_full_backward_hook(self.save_gradient)
        target_layer.register_forward_hook(self.save_activation)
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        # 2. Backward Pass (to get gradients)
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # 3. Generate Heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weight the activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU (only keep positive influence)
        cam = F.relu(cam)
        
        # Normalize 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.data.cpu().numpy()[0, 0], class_idx

# --- VISUALIZER ---
def generate_explanation(image_path):
    print(f"Generating explanation for: {image_path}")
    
    # 1. Load Model
    model = BrainTumorModel(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # EfficientNet-B0's last convolutional layer is in 'features' block 8
    target_layer = model.backbone.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # 2. Preprocess Image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    input_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True 
    
    # 3. Run Grad-CAM
    mask, class_idx = grad_cam(input_tensor)
    
    # 4. Overlay Heatmap
    # Resize mask to original image size
    heatmap = cv2.resize(mask, (raw_image.width, raw_image.height))
    
    # Convert to RGB heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # --- FIX WAS HERE ---
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    original = np.array(raw_image)
    superimposed = np.uint8(original * 0.6 + heatmap * 0.4)
    
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    prediction = CLASSES[class_idx]
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original MRI")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"AI Focus Area ({prediction.upper()})")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    print(f"Explanation generated for: {prediction.upper()}")

if __name__ == "__main__":
    # Test on a specific image (Update this path if needed!)
    # I am pointing to the glioma image we found earlier
    TEST_IMAGE = os.path.join("data", "raw", "MRI_images", "Testing", "pituitary", "Te-pi_0010.jpg")
    
    if os.path.exists(TEST_IMAGE):
        generate_explanation(TEST_IMAGE)
    else:
        print(f"Could not find image at: {TEST_IMAGE}")
        print("Please edit the 'TEST_IMAGE' path at the bottom of src/explain.py")