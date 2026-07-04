import torch
import torch.nn as nn

def extract_feature_maps(img_tensor):
    """
    Apply a simple convolution layer with 8 output channels, kernel size 3, padding 1
    to extract feature maps from a single-channel input image tensor.
    img_tensor shape: [1, 1, H, W]
    Returns the feature map tensor of shape [1, 8, H, W].
    """
    # TODO: Initialize Conv2D layer with 1 input channel, 8 output channels, kernel_size 3, padding 1
    # Hint: nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
    conv = nn.Identity()
    
    # TODO: Run forward pass
    out = conv(img_tensor)
        
    return out

if __name__ == "__main__":
    img = torch.randn(1, 1, 64, 64)
    out_maps = extract_feature_maps(img)
    
    print("--- Training Results ---")
    if out_maps.shape == (1, 8, 64, 64):
        print(f"Input Tensor Shape : {img.shape}")
        print(f"Feature Maps Shape : {out_maps.shape}")
    else:
        print("Feature map extraction not implemented yet.")
