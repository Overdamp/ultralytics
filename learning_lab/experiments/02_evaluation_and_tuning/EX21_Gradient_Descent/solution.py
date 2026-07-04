import numpy as np

def cost_function(x, y):
    return x**2 + 3 * y**2

def compute_gradient(x, y):
    return np.array([2 * x, 6 * y])

def gradient_descent_2d(start_pos, lr, epochs):
    """
    Run 2D Gradient Descent.
    """
    pos = np.array(start_pos, dtype=float)
    
    for _ in range(epochs):
        grad = compute_gradient(pos[0], pos[1])
        pos = pos - lr * grad
        
    cost = cost_function(pos[0], pos[1])
    return pos, cost

if __name__ == "__main__":
    start_pos = [8.0, 8.0]
    lr = 0.1
    epochs = 30
    
    final_pos, final_cost = gradient_descent_2d(start_pos, lr, epochs)
    
    print("--- Training Results ---")
    print(f"Final Position: x = {final_pos[0]:.6f}, y = {final_pos[1]:.6f}")
    print(f"Final Cost    : {final_cost:.6f}")
