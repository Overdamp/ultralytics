import torch
import torch.nn as nn

def create_mlp():
    """
    Create a PyTorch MLP Sequential model:
    - Input layer: 4 features -> 8 hidden features
    - ReLU activation
    - Output layer: 8 hidden features -> 2 output features
    """
    # TODO: Build sequential model
    # Hint: Use nn.Sequential, nn.Linear, nn.ReLU
    model = nn.Sequential()
    
    return model

if __name__ == "__main__":
    model = create_mlp()
    x = torch.tensor([[1.5, -0.5, 2.0, 0.0]])
    
    try:
        output = model(x)
        print("--- Training Results ---")
        print(f"Input Shape : {x.shape}")
        print(f"Output Shape: {output.shape}")
    except Exception as e:
        print("MLP architecture not implemented or error occurred:", e)
