import numpy as np

def mini_batch_gradient_descent(X, y, batch_size=2, lr=0.01, epochs=10, random_state=42):
    """
    Run Mini-Batch Gradient Descent for linear regression.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    m = len(X)
    w = 0.0
    b = 0.0
    
    for _ in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            B = len(X_batch)
            
            # Predict and calculate mini-batch error vector
            preds = w * X_batch + b
            error = preds - y_batch
            
            # Vectorized gradients
            dw = (1 / B) * np.dot(X_batch.T, error)[0, 0]
            db = (1 / B) * np.sum(error)
            
            # Update immediately
            w -= lr * dw
            b -= lr * db
            
    final_preds = w * X + b
    final_cost = (1 / (2 * m)) * np.sum((final_preds - y) ** 2)
    
    w_val = w.item() if hasattr(w, "item") else float(w)
    b_val = b.item() if hasattr(b, "item") else float(b)
    cost_val = final_cost.item() if hasattr(final_cost, "item") else float(final_cost)
    
    return w_val, b_val, cost_val

if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([[2.1], [3.9], [6.2], [8.0]])
    
    w, b, cost = mini_batch_gradient_descent(X, y, batch_size=2, lr=0.01, epochs=100)
    
    print("--- Training Results ---")
    print(f"Weight: {w:.4f} | Bias: {b:.4f} | Cost: {cost:.6f}")
