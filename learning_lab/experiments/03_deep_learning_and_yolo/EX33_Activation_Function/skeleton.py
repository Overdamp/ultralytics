import numpy as np

def sigmoid(x):
    # TODO: Implement sigmoid formula: 1 / (1 + exp(-x))
    return np.zeros_like(x)

def relu(x):
    # TODO: Implement ReLU formula: max(0, x)
    return np.zeros_like(x)

def leaky_relu(x, alpha=0.1):
    # TODO: Implement Leaky ReLU formula: max(alpha * x, x)
    return np.zeros_like(x)

def silu(x):
    # TODO: Implement SiLU (Swish) formula: x * sigmoid(x)
    return np.zeros_like(x)

if __name__ == "__main__":
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    
    sig_out = sigmoid(x)
    
    print("--- Training Results ---")
    if np.any(sig_out != 0.0):
        print(f"Input:      [{x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f}, {x[3]:.2f}, {x[4]:.2f}]")
        print(f"Sigmoid:    [{sigmoid(x)[0]:.4f}, {sigmoid(x)[1]:.4f}, {sigmoid(x)[2]:.4f}, {sigmoid(x)[3]:.4f}, {sigmoid(x)[4]:.4f}]")
        print(f"ReLU:       [{relu(x)[0]:.4f}, {relu(x)[1]:.4f}, {relu(x)[2]:.4f}, {relu(x)[3]:.4f}, {relu(x)[4]:.4f}]")
        print(f"Leaky ReLU: [{leaky_relu(x)[0]:.4f}, {leaky_relu(x)[1]:.4f}, {leaky_relu(x)[2]:.4f}, {leaky_relu(x)[3]:.4f}, {leaky_relu(x)[4]:.4f}]")
        print(f"SiLU:       [{silu(x)[0]:.4f}, {silu(x)[1]:.4f}, {silu(x)[2]:.4f}, {silu(x)[3]:.4f}, {silu(x)[4]:.4f}]")
    else:
        print("Activation functions logic not implemented yet.")
