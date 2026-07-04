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
    # TODO: Implement Forward Pass
    # z = w * x + b
    # a = sigmoid(z)
    # loss = (a - y)^2
    z = w * x + b
    a = 0.0
    loss = 0.0
    
    # TODO: Implement Backward Pass (Evaluate partial derivatives of the Chain Rule)
    # Hint: dL_da = 2 * (a - y)
    # Hint: da_dz = a * (1 - a)
    # Hint: dL_dw = dL_da * da_dz * x
    # Hint: dL_db = dL_da * da_dz * 1.0
    dL_dw = 0.0
    dL_db = 0.0
    
    return float(loss), float(a), float(dL_dw), float(dL_db)

if __name__ == "__main__":
    x = 1.5
    y = 2.0
    w = 0.8
    b = 0.2
    
    loss, pred, dw, db = manual_forward_backward(x, y, w, b)
    
    print("--- Training Results ---")
    if dw != 0.0 or db != 0.0:
        print(f"Prediction: {pred:.6f}")
        print(f"Loss      : {loss:.6f}")
        print(f"dL/dw     : {dw:.6f}")
        print(f"dL/db     : {db:.6f}")
    else:
        print("Backpropagation gradient calculation logic not implemented yet.")
