import numpy as np

def stochastic_gradient_descent(X, y, lr=0.01, epochs=10, random_state=42):
    """
    Run Stochastic Gradient Descent (SGD) for linear regression.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    m = len(X)
    w = 0.0
    b = 0.0
    
    for _ in range(epochs):
        # TODO: Shuffling indices randomly at start of epoch
        # Hint: Use np.random.permutation(m)
        indices = np.arange(m)
        
        for idx in indices:
            xi = X[idx]
            yi = y[idx]
            
            # TODO: Forward pass (single prediction) and single-sample error
            pred = 0.0
            error = 0.0
            
            # TODO: Immediate weight updates
            # Hint: w -= lr * error * xi
            # Hint: b -= lr * error
            pass
            
    final_preds = w * X + b
    final_cost = (1 / (2 * m)) * np.sum((final_preds - y) ** 2)
    
    # Use .item() to extract scalar values safely and avoid numpy deprecation warnings
    w_val = w.item() if hasattr(w, "item") else float(w)
    b_val = b.item() if hasattr(b, "item") else float(b)
    cost_val = final_cost.item() if hasattr(final_cost, "item") else float(final_cost)
    
    return w_val, b_val, cost_val

if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[2.1], [3.9], [6.2], [8.0]])
    
    w, b, cost = stochastic_gradient_descent(X, y, lr=0.01, epochs=100)
    
    print("--- Training Results ---")
    if w != 0.0 or b != 0.0:
        print(f"Weight: {w:.4f} | Bias: {b:.4f} | Cost: {cost:.6f}")
    else:
        print("SGD optimization logic not implemented yet.")
