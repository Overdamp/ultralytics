import numpy as np

def cost_ravine(x, y):
    return 0.5 * x**2 + 10.0 * y**2

def grad_ravine(x, y):
    return np.array([x, 20.0 * y])

def adam_gradient_descent(start_pos, lr=0.15, beta1=0.9, beta2=0.999, eps=1e-8, epochs=50):
    """
    Run 2D Gradient Descent using Adam.
    """
    pos = np.array(start_pos, dtype=float)
    # TODO: Initialize moments m and v as zeros of shape (2,)
    m = np.zeros(2)
    v = np.zeros(2)
    
    # Note: loop index t starts at 1 for correct power calculations (t = 1 to epochs)
    for t in range(1, epochs + 1):
        grad = grad_ravine(pos[0], pos[1])
        
        # TODO: Update first moment m
        # Hint: m = beta1 * m + (1.0 - beta1) * grad
        
        # TODO: Update second moment v
        # Hint: v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        
        # TODO: Bias correction
        # Hint: m_hat = m / (1.0 - beta1 ** t)
        # Hint: v_hat = v / (1.0 - beta2 ** t)
        m_hat = m
        v_hat = v
        
        # TODO: Update parameters
        # Hint: pos -= (lr / (sqrt(v_hat) + eps)) * m_hat
        pass
        
    cost = cost_ravine(pos[0], pos[1])
    return pos, cost

if __name__ == "__main__":
    start_pos = [8.0, 4.0]
    lr = 0.25
    beta1 = 0.9
    beta2 = 0.999
    epochs = 50
    
    final_pos, final_cost = adam_gradient_descent(start_pos, lr, beta1, beta2, epochs=epochs)
    
    print("--- Training Results ---")
    if final_cost < cost_ravine(8.0, 4.0):
        print(f"Final Position: x = {final_pos[0]:.6f}, y = {final_pos[1]:.6f}")
        print(f"Final Cost    : {final_cost:.6f}")
    else:
        print("Adam gradient descent optimization not implemented yet.")
