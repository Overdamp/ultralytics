import numpy as np

def sgd_weight_decay_update(w, grad, lr, weight_decay):
    """
    Perform parameter update using SGD with Weight Decay.
    w: weight array/matrix (numpy array)
    grad: raw loss gradient array/matrix (numpy array)
    lr: learning rate (float)
    weight_decay: weight decay parameter (float)
    """
    # TODO: Calculate updated weights with weight decay shrinkage
    # Hint: w_new = w * (1.0 - lr * weight_decay) - lr * grad
    w_updated = w.copy()
    
    return w_updated

if __name__ == "__main__":
    w = np.array([1.5, -2.0, 0.5])
    grad = np.array([0.1, -0.3, 0.05])
    lr = 0.1
    weight_decay = 0.005
    
    w_new = sgd_weight_decay_update(w, grad, lr, weight_decay)
    
    print("--- Training Results ---")
    if not np.array_equal(w_new, w):
        print(f"Updated weights: [{w_new[0]:.6f}, {w_new[1]:.6f}, {w_new[2]:.6f}]")
    else:
        print("Weight decay updates not implemented yet.")
