import numpy as np

def manual_forward_backward(x, y, w, b):
    """
    Perform a single forward and backward pass for a single sigmoid neuron.
    Returns:
    - loss (float): squared error
    - prediction (float): sigmoid output
    - dL_dw (float): gradient of loss wrt weight w
    - dL_db (float): gradient of loss wrt bias b
    """
    # 1. Forward Pass
    z = w * x + b
    a = 1.0 / (1.0 + np.exp(-z))
    loss = (a - y) ** 2
    
    # 2. Backward Pass (using Chain Rule)
    dL_da = 2.0 * (a - y)
    da_dz = a * (1.0 - a)
    
    dL_dw = dL_da * da_dz * x
    dL_db = dL_da * da_dz * 1.0
    
    return float(loss), float(a), float(dL_dw), float(dL_db)

if __name__ == "__main__":
    x = 1.5
    y = 2.0
    w = 0.8
    b = 0.2
    
    loss, pred, dw, db = manual_forward_backward(x, y, w, b)
    
    print("--- Training Results ---")
    print(f"Prediction: {pred:.6f}")
    print(f"Loss      : {loss:.6f}")
    print(f"dL/dw     : {dw:.6f}")
    print(f"dL/db     : {db:.6f}")
