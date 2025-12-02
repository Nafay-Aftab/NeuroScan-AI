import torch.nn as nn
from torchvision import models

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(BrainTumorModel, self).__init__()
        
        # 1. Load the pre-trained EfficientNet-B0 model
        # "DEFAULT" weights means it uses the best available ImageNet weights
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # 2. Freeze the early layers (Optional but good for small datasets)
        # This prevents the model from "forgetting" how to see basic shapes
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        # 3. Replace the Head (Classifier)
        # EfficientNet's classifier is a Sequential block. 
        # Layer [1] is the final Linear layer we need to change.
        # We get the input number of features from the existing layer before deleting it.
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)