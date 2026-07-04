import torch
import torch.nn as nn

def create_mlp():
    """
    Create a PyTorch MLP Sequential model:
    - Input layer: 4 features -> 8 hidden features
    - ReLU activation
    - Output layer: 8 hidden features -> 2 output features
    """
    model = nn.Sequential(
        nn.Linear(in_features=4, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=2)
    )
    return model

if __name__ == "__main__":
    model = create_mlp()
    x = torch.tensor([[1.5, -0.5, 2.0, 0.0]])
    output = model(x)
    
    print("--- Training Results ---")
    print(f"Input Shape : {x.shape}")
    print(f"Output Shape: {output.shape}")
