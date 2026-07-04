import torch
import torch.nn as nn
import torchvision.models as models

def count_parameters(model):
    """
    Count total and trainable parameters in a PyTorch model.
    """
    # TODO: Implement total and trainable parameters counter
    total_params = 0
    trainable_params = 0
    return total_params, trainable_params

def setup_transfer_learning(model, num_classes=10):
    """
    Freeze all backbone layers and replace the final fully connected layer (model.fc)
    with a new linear classifier layer for the specified number of classes.
    """
    # TODO: Loop over parameters, freeze all (requires_grad = False)
    # TODO: Replace model.fc with a new linear layer: nn.Linear(in_features, num_classes)
    # Hint: Get in_features from model.fc.in_features
    pass
    
    return model

if __name__ == "__main__":
    # Load ResNet18 model
    model = models.resnet18()
    
    total_init, trainable_init = count_parameters(model)
    
    model = setup_transfer_learning(model, num_classes=5)
    
    total_final, trainable_final = count_parameters(model)
    
    print("--- Training Results ---")
    if trainable_final != 0 and trainable_final != trainable_init:
        print(f"Initial Trainable Parameters: {trainable_init:,}")
        print(f"Final Trainable Parameters  : {trainable_final:,} (expected: 2,565)")
        print(f"Output Layer Weight Shape   : {list(model.fc.weight.shape)} (expected: [5, 512])")
    else:
        print("Transfer learning freezing setup not implemented yet.")
