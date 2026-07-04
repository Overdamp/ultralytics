import numpy as np

def cost_function(x, y):
    return x**2 + 3 * y**2

def compute_gradient(x, y):
    # TODO: Compute partial derivatives for f(x, y) = x^2 + 3 * y^2
    # Hint: df/dx = 2 * x, df/dy = 6 * y
    df_dx = 0.0
    df_dy = 0.0
    return np.array([df_dx, df_dy])

def gradient_descent_2d(start_pos, lr, epochs):
    """
    Run 2D Gradient Descent.
    """
    pos = np.array(start_pos, dtype=float)
    
    # TODO: Loop over epochs, evaluate gradient and update position
    # Hint: pos = pos - lr * grad
    for _ in range(epochs):
        pass
        
    cost = cost_function(pos[0], pos[1])
    return pos, cost

if __name__ == "__main__":
    start_pos = [8.0, 8.0]
    lr = 0.1
    epochs = 30
    
    final_pos, final_cost = gradient_descent_2d(start_pos, lr, epochs)
    
    print("--- Training Results ---")
    if final_cost != cost_function(8.0, 8.0):
        print(f"Final Position: x = {final_pos[0]:.6f}, y = {final_pos[1]:.6f}")
        print(f"Final Cost    : {final_cost:.6f}")
    else:
        print("Gradient descent update logic not implemented yet.")
