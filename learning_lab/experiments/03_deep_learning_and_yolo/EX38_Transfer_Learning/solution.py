import torch
import torch.nn as nn
import torchvision.models as models

def count_parameters(model):
    """
    Count total and trainable parameters in a PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def setup_transfer_learning(model, num_classes=10):
    """
    Freeze all backbone layers and replace the final fully connected layer (model.fc)
    with a new linear classifier layer for the specified number of classes.
    """
    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Replace the output classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Load ResNet18 model
    model = models.resnet18()
    
    total_init, trainable_init = count_parameters(model)
    
    # Setup model for transfer learning with 5 classes
    model = setup_transfer_learning(model, num_classes=5)
    
    total_final, trainable_final = count_parameters(model)
    
    print("--- Training Results ---")
    print(f"Initial Trainable Parameters: {trainable_init:,}")
    print(f"Final Trainable Parameters  : {trainable_final:,} (expected: 2,565)")
    print(f"Output Layer Weight Shape   : {list(model.fc.weight.shape)} (expected: [5, 512])")
