import numpy as np

class SupportVectorMachine:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        """
        Fit SVM using Soft-Margin subgradient descent.
        """
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Check soft-margin constraint condition
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Regularize weights
                    self.w -= self.lr * (2 * (1 / self.epochs) * self.w)
                else:
                    # Violator: adjust weights and bias
                    self.w -= self.lr * (2 * (1 / self.epochs) * self.w - self.C * y[idx] * x_i)
                    self.b += self.lr * self.C * y[idx]

    def predict(self, X):
        """
        Predict class labels (-1 or +1).
        """
        return np.sign(np.dot(X, self.w) + self.b)

if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [2.0, 1.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([-1, -1, 1, 1])
    
    model = SupportVectorMachine(C=10.0, learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    print("--- Training Results ---")
    print(f"w: {model.w} | b: {model.b:.4f}")
    
    test_X = np.array([
        [1.5, 1.5],
        [5.5, 5.5]
    ])
    preds = model.predict(test_X)
    print(f"\nPrediction for [1.5, 1.5]: Class {int(preds[0])} (expected: -1)")
    print(f"Prediction for [5.5, 5.5]: Class {int(preds[1])} (expected: 1)")
