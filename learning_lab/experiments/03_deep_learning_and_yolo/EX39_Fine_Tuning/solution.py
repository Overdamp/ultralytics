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

def setup_partial_fine_tuning(model, num_classes=10, freeze_list=None):
    """
    1. Freeze parameters if their name contains any string in freeze_list.
    2. Swap the final classification head (model.fc) with a new linear layer 
       for the specified number of classes (which is always trainable).
    """
    if freeze_list is None:
        freeze_list = []
        
    for name, param in model.named_parameters():
        if any(match_str in name for match_str in freeze_list):
            param.requires_grad = False
        else:
            param.requires_grad = True
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Load ResNet18 model
    model = models.resnet18()
    
    total_init, trainable_init = count_parameters(model)
    
    # Setup partial fine-tuning: freeze early layers (conv1, bn1, layer1)
    # and replace head with a 10-class classifier
    model = setup_partial_fine_tuning(model, num_classes=10, freeze_list=['conv1', 'bn1', 'layer1'])
    
    total_final, trainable_final = count_parameters(model)
    
    print("--- Training Results ---")
    print(f"Initial Trainable Parameters: {trainable_init:,}")
    print(f"Final Trainable Parameters  : {trainable_final:,}")
