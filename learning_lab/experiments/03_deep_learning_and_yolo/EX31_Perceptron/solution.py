import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=0.1):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0
        self.lr = lr
        
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                error = yi - y_pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR Gate
    
    p = Perceptron(input_dim=2)
    p.train(X, y, epochs=10)
    
    print("--- Training Results ---")
    print(f"Weights: [{p.weights[0]:.2f}, {p.weights[1]:.2f}]")
    print(f"Bias   : {p.bias:.2f}")
    print(f"Predict [1, 0] -> {p.predict([1, 0])} (expected: 1)")
    print(f"Predict [0, 0] -> {p.predict([0, 0])} (expected: 0)")
