import numpy as np

def cost_ravine(x, y):
    return 0.5 * x**2 + 10.0 * y**2

def grad_ravine(x, y):
    return np.array([x, 20.0 * y])

def momentum_gradient_descent(start_pos, lr=0.08, beta=0.9, epochs=40):
    """
    Run 2D Gradient Descent with Momentum.
    """
    pos = np.array(start_pos, dtype=float)
    v = np.zeros(2)
    
    for _ in range(epochs):
        grad = grad_ravine(pos[0], pos[1])
        v = beta * v + lr * grad
        pos -= v
        
    cost = cost_ravine(pos[0], pos[1])
    return pos, cost

if __name__ == "__main__":
    start_pos = [8.0, 4.0]
    lr = 0.08
    beta = 0.9
    epochs = 40
    
    final_pos, final_cost = momentum_gradient_descent(start_pos, lr, beta, epochs)
    
    print("--- Training Results ---")
    print(f"Final Position: x = {final_pos[0]:.6f}, y = {final_pos[1]:.6f}")
    print(f"Final Cost    : {final_cost:.6f}")
