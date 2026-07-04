import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
        
    def predict(self, x):
        # TODO: Calculate weighted sum z = w * x + bias, return 1 if z >= 0 else 0
        # Hint: Use np.dot(x, self.weights)
        z = 0.0
        return 0

    def train(self, X, y, epochs=10):
        # TODO: Implement the Perceptron training loop.
        # For each epoch:
        #   For each sample (xi, yi) in (X, y):
        #     Compute prediction y_pred.
        #     Compute error: yi - y_pred.
        #     If error is not zero, update weights and bias:
        #       self.weights += self.lr * error * xi
        #       self.bias += self.lr * error
        pass

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR Gate
    
    p = Perceptron(input_dim=2)
    p.train(X, y, epochs=10)
    
    print("--- Training Results ---")
    test_pred = p.predict([1, 0])
    if test_pred != 0 or np.any(p.weights != 0.0):
        print(f"Weights: [{p.weights[0]:.2f}, {p.weights[1]:.2f}]")
        print(f"Bias   : {p.bias:.2f}")
        print(f"Predict [1, 0] -> {p.predict([1, 0])} (expected: 1)")
        print(f"Predict [0, 0] -> {p.predict([0, 0])} (expected: 0)")
    else:
        print("Perceptron classification logic not implemented yet.")
